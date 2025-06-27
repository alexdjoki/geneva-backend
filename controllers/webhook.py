from flask import Blueprint, request, jsonify
import os
from svix.webhooks import Webhook, WebhookVerificationError
from models.auth import AccessKey
from models.transactions import Transactions
from dateutil.relativedelta import relativedelta
from app import db
from datetime import datetime
from sqlalchemy import func, or_
import re

webhook_bp = Blueprint('webhook', __name__)
WEBHOOK_SECRET = os.getenv('CRISSMINT_API_KEY')
def find_access_key_by_email(email: str):
    """Find access key by email with proper normalized comparison."""
    if not email:
        return None

    normalized_email = normalize_email(email)

    # Search using normalized versions of both sides
    token = db.session.query(AccessKey).filter(
        func.replace(func.lower(AccessKey.email), '.', '') == func.replace(func.lower(normalized_email), '.', '')
    ).first()

    # If not found, try more flexible matching
    if not token:
        logger.debug(f"No normalized match found for {email}, trying partial match")
        username, domain = normalized_email.split('@')
        search_pattern = f"%{username}%{domain}"

        token = db.session.query(AccessKey).filter(
            func.replace(func.lower(AccessKey.email), '.', '').like(search_pattern)
        ).first()

    return token

def normalize_email(email: str) -> str:
    """Normalize email by handling Gmail-specific cases and converting to lowercase."""
    if not email:
        return None

    email = email.lower().strip()

    # Handle Gmail addresses
    if '@gmail.com' in email:
        username, domain = email.split('@')
        username = username.replace('.', '')  # Remove all dots
        username = re.sub(r'\+.*', '', username)  # Remove + suffixes
        return f"{username}@{domain}"

    return email

@webhook_bp.route('/', methods=['POST'])
def handle_webhook():
    """Handle incoming webhook events from Crossmint."""
    # Validate required headers first
    required_headers = ["svix-id", "svix-timestamp", "svix-signature"]
    if not all(h in request.headers for h in required_headers):
        print("Missing required Svix headers")
        return jsonify({"error": "Missing required headers"}), 400

    # Webhook verification
    try:
        headers = {
            "svix-id": request.headers["svix-id"],
            "svix-timestamp": request.headers["svix-timestamp"],
            "svix-signature": request.headers["svix-signature"],
        }
        wh = Webhook(WEBHOOK_SECRET)
        payload = wh.verify(request.get_data(as_text=True), headers)
    except Exception as e:
        print(f"Webhook verification failed: {str(e)}")
        return jsonify({"error": "Invalid webhook signature"}), 401

    try:
        event_type = payload.get("type")
        data = payload.get("data", {})

        # Log basic event info
        print(f"Processing {event_type} event")

        # Only process delivery completed events
        if event_type != "orders.delivery.completed":
            print(f"Ignoring non-delivery event: {event_type}")
            return jsonify({"status": "ignored"}), 200

        # Validate required fields
        if not (order_id := data.get('orderId')):
            print("Missing orderId in payload")
            return jsonify({"error": "Missing orderId"}), 400

        line_items = data.get('lineItems', [])
        if not line_items:
            print("No line items found in payload")
            return jsonify({"error": "No line items found"}), 400

        # Extract recipient information with better safety checks
        try:
            recipient = line_items[0].get('delivery', {}).get('recipient', {})
            if not (email := recipient.get('email')):
                print("Missing recipient email in payload")
                return jsonify({"error": "Missing recipient email"}), 400
        except (AttributeError, IndexError) as e:
            print(f"Malformed recipient data: {str(e)}")
            return jsonify({"error": "Invalid recipient data"}), 400

        # Find access key with proper error handling
        token = find_access_key_by_email(email)
        if not token:
            print(f"No access key found for email: {email}")
            return jsonify({"error": "No access key found for this email"}), 404

        # Process payment information with better formatting
        payment = data.get('payment', {})
        total_price = data.get('quote', {}).get('totalPrice', {})
        amount = f"{total_price.get('amount', '0')} {total_price.get('currency', 'unknown')}"

        # Transaction processing with proper session management
        try:
            # Check if transaction already exists
            transaction = Transactions.query.filter_by(order_id=order_id).first()
            transaction = Transactions(
                wallet_address=recipient.get('walletAddress'),
                email=token.email,
                amount=amount,
                userid=token.device_id,
                access_key=token.access_key,
                created_at=datetime.utcnow(),
                order_id=order_id,
                status="completed",
            )
            db.session.add(transaction)
            # Update token validity with proper date handling
            current_time = datetime.utcnow()
            if token.valid_date and token.valid_date > current_time:
                token.valid_date += relativedelta(months=1)
            else:
                token.valid_date = current_time + relativedelta(months=1)

            print(f"Updated token validity for {email} until {token.valid_date}")
            db.session.commit()

            return jsonify({"status": "success"}), 200

        except Exception as db_error:
            print(db_error)
            db.session.rollback()
            return jsonify({"error": "Transaction processing failed"}), 500

    except Exception as e:
        print(e)
        return jsonify({"error": "Internal server error"}), 500