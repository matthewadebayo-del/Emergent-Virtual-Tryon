<?php
/**
 * Plugin Name: VirtualFit Try-On
 * Description: Virtual try-on integration for WooCommerce
 * Version: 1.0.0
 * Author: VirtualFit
 */

if (!defined('ABSPATH')) {
    exit;
}

class VirtualFitWooCommerce {
    
    public function __construct() {
        add_action('init', array($this, 'init'));
        add_action('wp_enqueue_scripts', array($this, 'enqueue_scripts'));
        add_action('woocommerce_single_product_summary', array($this, 'add_tryon_button'), 25);
        add_action('wp_ajax_virtualfit_process', array($this, 'ajax_process_tryon'));
        add_action('wp_ajax_nopriv_virtualfit_process', array($this, 'ajax_process_tryon'));
        add_action('admin_menu', array($this, 'admin_menu'));
    }
    
    public function init() {
        // Plugin initialization
    }
    
    public function enqueue_scripts() {
        if (is_product()) {
            wp_enqueue_script('virtualfit-sdk', plugin_dir_url(__FILE__) . '../javascript/virtualfit-sdk.js', array(), '1.0.0', true);
            wp_enqueue_script('virtualfit-woo', plugin_dir_url(__FILE__) . 'virtualfit-woo.js', array('virtualfit-sdk'), '1.0.0', true);
            wp_enqueue_style('virtualfit-style', plugin_dir_url(__FILE__) . 'virtualfit-style.css', array(), '1.0.0');
            
            wp_localize_script('virtualfit-woo', 'virtualfit_ajax', array(
                'ajax_url' => admin_url('admin-ajax.php'),
                'nonce' => wp_create_nonce('virtualfit_nonce'),
                'api_key' => get_option('virtualfit_api_key', ''),
                'base_url' => get_option('virtualfit_base_url', 'https://api.virtualfit.com/api/v1')
            ));
        }
    }
    
    public function add_tryon_button() {
        global $product;
        
        if (!$this->is_tryon_eligible($product)) {
            return;
        }
        
        $product_data = $this->get_product_data($product);
        
        echo '<div class="virtualfit-container">';
        echo '<button type="button" class="button virtualfit-btn" data-product="' . esc_attr(json_encode($product_data)) . '">';
        echo '👕 Try On Virtual';
        echo '</button>';
        echo '</div>';
    }
    
    private function is_tryon_eligible($product) {
        $categories = wp_get_post_terms($product->get_id(), 'product_cat', array('fields' => 'slugs'));
        $eligible_categories = array('clothing', 'shirts', 'tops', 'dresses', 'pants', 'jeans');
        
        return !empty(array_intersect($categories, $eligible_categories));
    }
    
    private function get_product_data($product) {
        $categories = wp_get_post_terms($product->get_id(), 'product_cat', array('fields' => 'names'));
        $category = $this->map_category($categories);
        
        $attributes = $product->get_attributes();
        $color = $this->extract_attribute($attributes, 'color');
        $size = $this->extract_attribute($attributes, 'size');
        
        return array(
            'id' => $product->get_id(),
            'name' => $product->get_name(),
            'category' => $category,
            'imageUrl' => wp_get_attachment_image_src($product->get_image_id(), 'full')[0],
            'color' => $color,
            'size' => $size
        );
    }
    
    private function map_category($categories) {
        $category_map = array(
            'shirts' => 'TOP',
            'tops' => 'TOP',
            't-shirts' => 'TOP',
            'pants' => 'BOTTOM',
            'jeans' => 'BOTTOM',
            'dresses' => 'DRESS',
            'shoes' => 'SHOES',
            'jackets' => 'OUTERWEAR'
        );
        
        foreach ($categories as $cat) {
            $slug = strtolower($cat);
            if (isset($category_map[$slug])) {
                return $category_map[$slug];
            }
        }
        
        return 'TOP';
    }
    
    private function extract_attribute($attributes, $name) {
        foreach ($attributes as $attribute) {
            if (strpos(strtolower($attribute->get_name()), $name) !== false) {
                return $attribute->get_options()[0] ?? 'default';
            }
        }
        return 'default';
    }
    
    public function ajax_process_tryon() {
        check_ajax_referer('virtualfit_nonce', 'nonce');
        
        $customer_image = $_FILES['customer_image'] ?? null;
        $product_data = json_decode(stripslashes($_POST['product_data']), true);
        
        if (!$customer_image || !$product_data) {
            wp_die('Invalid request');
        }
        
        try {
            // Upload customer image
            $upload = wp_handle_upload($customer_image, array('test_form' => false));
            if (isset($upload['error'])) {
                throw new Exception($upload['error']);
            }
            
            // Process with VirtualFit API
            $result = $this->call_virtualfit_api($upload['url'], $product_data);
            
            wp_send_json_success($result);
            
        } catch (Exception $e) {
            wp_send_json_error($e->getMessage());
        }
    }
    
    private function call_virtualfit_api($customer_image_url, $product_data) {
        $api_key = get_option('virtualfit_api_key');
        $base_url = get_option('virtualfit_base_url', 'https://api.virtualfit.com/api/v1');
        
        $payload = array(
            'api_key' => $api_key,
            'customer_image_url' => $customer_image_url,
            'garment_image_url' => $product_data['imageUrl'],
            'product_info' => $product_data
        );
        
        $response = wp_remote_post($base_url . '/tryon/sync', array(
            'headers' => array('Content-Type' => 'application/json'),
            'body' => json_encode($payload),
            'timeout' => 60
        ));
        
        if (is_wp_error($response)) {
            throw new Exception($response->get_error_message());
        }
        
        $body = wp_remote_retrieve_body($response);
        return json_decode($body, true);
    }
    
    public function admin_menu() {
        add_options_page(
            'VirtualFit Settings',
            'VirtualFit',
            'manage_options',
            'virtualfit-settings',
            array($this, 'admin_page')
        );
    }
    
    public function admin_page() {
        if (isset($_POST['submit'])) {
            update_option('virtualfit_api_key', sanitize_text_field($_POST['api_key']));
            update_option('virtualfit_base_url', esc_url_raw($_POST['base_url']));
            echo '<div class="notice notice-success"><p>Settings saved!</p></div>';
        }
        
        $api_key = get_option('virtualfit_api_key', '');
        $base_url = get_option('virtualfit_base_url', 'https://api.virtualfit.com/api/v1');
        
        ?>
        <div class="wrap">
            <h1>VirtualFit Settings</h1>
            <form method="post">
                <table class="form-table">
                    <tr>
                        <th scope="row">API Key</th>
                        <td><input type="text" name="api_key" value="<?php echo esc_attr($api_key); ?>" class="regular-text" /></td>
                    </tr>
                    <tr>
                        <th scope="row">Base URL</th>
                        <td><input type="url" name="base_url" value="<?php echo esc_attr($base_url); ?>" class="regular-text" /></td>
                    </tr>
                </table>
                <?php submit_button(); ?>
            </form>
        </div>
        <?php
    }
}

new VirtualFitWooCommerce();