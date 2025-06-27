CREATE TABLE `dashboard` (
	`id` INT(11) NOT NULL AUTO_INCREMENT,
	`monthly_active` INT(11) NULL DEFAULT '0',
	`token_count` INT(11) NULL DEFAULT '0',
	`total_input_tokens` INT(11) NULL DEFAULT '0',
	`total_output_tokens` INT(11) NULL DEFAULT '0',
	`cost_to_date` INT(11) NULL DEFAULT '0',
	`chat_count` INT(11) NULL DEFAULT '0',
	`product_count` INT(11) NOT NULL DEFAULT '0',
	`created_at` TIMESTAMP NULL DEFAULT NULL,
	`updated_at` TIMESTAMP NULL DEFAULT NULL,
	PRIMARY KEY (`id`) USING BTREE
)
COLLATE='utf8mb4_general_ci'
ENGINE=InnoDB
AUTO_INCREMENT=2
;

INSERT INTO `dashboard` (`id`, `monthly_active`, `token_count`, `total_input_tokens`, `total_output_tokens`, `cost_to_date`, `chat_count`, `product_count`, `created_at`, `updated_at`) VALUES (1, 1336, 4575, 2074800, 1333800, 357, 76, 24, '2025-06-17 00:39:59', '2025-06-17 00:40:00');

