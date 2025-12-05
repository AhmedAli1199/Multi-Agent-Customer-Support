"""
Product search and information tools for Knowledge Agent.
Allows agents to search product catalog and provide product information.
"""
import json
from typing import List, Dict, Optional
from langchain.tools import tool
from pathlib import Path

# Load product catalog and company info
PRODUCT_CATALOG_FILE = Path(__file__).parent.parent.parent / "data" / "product_catalog.json"
COMPANY_INFO_FILE = Path(__file__).parent.parent.parent / "data" / "company_info.json"
COMPANY_FAQS_FILE = Path(__file__).parent.parent.parent / "data" / "company_faqs.json"

# Cache the data
_product_catalog = None
_company_info = None
_company_faqs = None


def load_product_catalog() -> Dict:
    """Load product catalog from JSON file"""
    global _product_catalog
    if _product_catalog is None:
        with open(PRODUCT_CATALOG_FILE, 'r', encoding='utf-8') as f:
            _product_catalog = json.load(f)
    return _product_catalog


def load_company_info() -> Dict:
    """Load company information from JSON file"""
    global _company_info
    if _company_info is None:
        with open(COMPANY_INFO_FILE, 'r', encoding='utf-8') as f:
            _company_info = json.load(f)
    return _company_info


def load_company_faqs() -> List[Dict]:
    """Load company FAQs from JSON file"""
    global _company_faqs
    if _company_faqs is None:
        with open(COMPANY_FAQS_FILE, 'r', encoding='utf-8') as f:
            _company_faqs = json.load(f)
    return _company_faqs


@tool
def search_products(query: str, category: Optional[str] = None, max_results: int = 5) -> str:
    """
    Search the TechGear product catalog by keyword or category.

    Args:
        query: Search query (e.g., "laptop", "wireless headphones", "gaming")
        category: Optional category filter (e.g., "Laptops & Computers", "Audio & Headphones")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted list of matching products with details
    """
    catalog = load_product_catalog()
    products = catalog["products"]

    query_lower = query.lower()
    matching_products = []

    for product in products:
        score = 0

        # Search in name, description, category, brand
        if query_lower in product["name"].lower():
            score += 10
        if query_lower in product["description"].lower():
            score += 5
        if query_lower in product["category"].lower():
            score += 3
        if query_lower in product["brand"].lower():
            score += 3

        # Check keywords in query
        for word in query_lower.split():
            if len(word) > 3:  # Skip short words
                if word in product["name"].lower():
                    score += 2
                if word in product["description"].lower():
                    score += 1

        # Category filter
        if category and category.lower() not in product["category"].lower():
            continue

        if score > 0:
            matching_products.append((score, product))

    # Sort by score
    matching_products.sort(key=lambda x: x[0], reverse=True)

    if not matching_products:
        return f"No products found matching '{query}'. Try browsing our categories or contact support for assistance."

    # Format results
    results = []
    for score, product in matching_products[:max_results]:
        stock_status = "✓ In Stock" if product["in_stock"] else f"✗ Out of Stock (Restocking: {product.get('expected_restock', 'TBA')})"

        result = f"""**{product['name']}** (ID: {product['product_id']})
- Price: ${product['price']}
- {stock_status}
- Category: {product['category']}
- Rating: {product['rating']}/5.0 ({product['reviews_count']} reviews)
- {product['description']}
"""
        results.append(result)

    header = f"Found {len(matching_products)} product(s) matching '{query}':\n\n"
    return header + "\n---\n".join(results)


@tool
def get_product_details(product_id: str) -> str:
    """
    Get detailed specifications and information for a specific product.

    Args:
        product_id: Product ID (e.g., "LAPTOP-001", "PHONE-001")

    Returns:
        Complete product details including specifications
    """
    catalog = load_product_catalog()
    products = catalog["products"]

    product = next((p for p in products if p["product_id"] == product_id), None)

    if not product:
        return f"Product {product_id} not found. Please check the product ID and try again."

    stock_status = "In Stock" if product["in_stock"] else f"Out of Stock (Expected: {product.get('expected_restock', 'TBA')})"
    stock_qty = f"({product['stock_quantity']} available)" if product["in_stock"] else ""

    specs = "\n".join(f"  - {k.replace('_', ' ').title()}: {v}" for k, v in product.get("specifications", {}).items())

    details = f"""**{product['name']}**

**Product ID**: {product_id}
**Brand**: {product['brand']}
**Category**: {product['category']}

**Price**: ${product['price']}
**Availability**: {stock_status} {stock_qty}

**Description**:
{product['description']}

**Specifications**:
{specs}

**Warranty**: {product.get('warranty', 'Standard 2-year warranty')}
**Rating**: {product['rating']}/5.0 ({product['reviews_count']} customer reviews)

**Purchase Options**:
- Free shipping on orders over $50
- 30-day return policy
- 2-year warranty included
- Premium members get FREE 2-day shipping
"""
    return details


@tool
def check_product_availability(product_id: str) -> str:
    """
    Check if a product is currently in stock and get availability details.

    Args:
        product_id: Product ID to check

    Returns:
        Stock status and availability information
    """
    catalog = load_product_catalog()
    products = catalog["products"]

    product = next((p for p in products if p["product_id"] == product_id), None)

    if not product:
        return f"Product {product_id} not found."

    if product["in_stock"]:
        qty = product["stock_quantity"]
        if qty > 50:
            stock_msg = "In Stock - Ready to ship immediately"
        elif qty > 10:
            stock_msg = f"In Stock - Limited quantity ({qty} available)"
        else:
            stock_msg = f"Low Stock - Only {qty} left!"

        return f"""{product['name']} (ID: {product_id})
Status: {stock_msg}

Order now to get it shipped within 1 business day!
- Standard Shipping: 5-7 days ($5.99, FREE over $50)
- Express Shipping: 2-3 days ($14.99)
- Overnight Shipping: Next day ($29.99)
"""
    else:
        restock = product.get("expected_restock", "To be announced")
        return f"""{product['name']} (ID: {product_id})
Status: Currently Out of Stock
Expected Restock: {restock}

Would you like me to:
1. Notify you when back in stock
2. Suggest similar in-stock alternatives
3. Process a pre-order (if available)
"""


@tool
def get_product_categories() -> str:
    """
    Get a list of all product categories available at TechGear Electronics.

    Returns:
        List of product categories with descriptions
    """
    company_info = load_company_info()
    categories = company_info["product_categories"]

    result = "**TechGear Electronics Product Categories**:\n\n"
    for i, category in enumerate(categories, 1):
        result += f"{i}. {category}\n"

    result += "\nTo browse products in a category, just ask me about it! For example: 'Show me laptops' or 'What smart home devices do you have?'"

    return result


@tool
def compare_products(product_id1: str, product_id2: str) -> str:
    """
    Compare two products side-by-side to help customer decide.

    Args:
        product_id1: First product ID
        product_id2: Second product ID

    Returns:
        Side-by-side comparison of the two products
    """
    catalog = load_product_catalog()
    products = catalog["products"]

    p1 = next((p for p in products if p["product_id"] == product_id1), None)
    p2 = next((p for p in products if p["product_id"] == product_id2), None)

    if not p1:
        return f"Product {product_id1} not found."
    if not p2:
        return f"Product {product_id2} not found."

    comparison = f"""**Product Comparison**: {p1['name']} vs {p2['name']}

| Feature | {p1['name']} | {p2['name']} |
|---------|--------------|--------------|
| **Price** | ${p1['price']} | ${p2['price']} |
| **Brand** | {p1['brand']} | {p2['brand']} |
| **Category** | {p1['category']} | {p2['category']} |
| **Rating** | {p1['rating']}/5.0 ({p1['reviews_count']} reviews) | {p2['rating']}/5.0 ({p2['reviews_count']} reviews) |
| **Availability** | {"In Stock" if p1['in_stock'] else "Out of Stock"} | {"In Stock" if p2['in_stock'] else "Out of Stock"} |

**{p1['name']}**:
{p1['description']}

**{p2['name']}**:
{p2['description']}

**Price Difference**: ${abs(p1['price'] - p2['price']):.2f}

Would you like more detailed specifications for either product?
"""
    return comparison


@tool
def get_company_info(info_type: str = "general") -> str:
    """
    Get TechGear Electronics company information, policies, and contact details.

    Args:
        info_type: Type of information (general, contact, shipping, returns, warranty, payment)

    Returns:
        Requested company information
    """
    company = load_company_info()

    if info_type == "general":
        return f"""{company['company_name']} - {company['tagline']}

Founded in {company['founded']}, headquartered in {company['headquarters']}.

{company['description']}

**Our Values**:
{chr(10).join('- ' + v for v in company['values'])}

For more specific information, ask about:
- Contact information
- Shipping policies
- Return policies
- Warranty information
- Payment methods
"""

    elif info_type == "contact":
        contact = company['contact']
        return f"""**Contact TechGear Electronics**:

📞 **Phone**: {contact['customer_support_phone']}
  - {contact['support_hours']['phone']}

📧 **Email**: {contact['customer_support_email']}
  - {contact['support_hours']['email']}

💬 **Live Chat**: {contact['support_hours']['chat']}

🌐 **Website**: {contact['website']}

**Social Media**:
- Twitter: {contact['social_media']['twitter']}
- Facebook: {contact['social_media']['facebook']}
- Instagram: {contact['social_media']['instagram']}
"""

    elif info_type == "shipping":
        shipping = company['policies']['shipping']
        return f"""**TechGear Shipping Information**:

**FREE Shipping**: On orders ${shipping['free_shipping_threshold']} or more!

**Standard Shipping**:
- Cost: {shipping['standard_shipping']['cost']}
- Delivery: {shipping['standard_shipping']['delivery_time']}
- Tracking: {shipping['standard_shipping']['tracking']}

**Express Shipping**:
- Cost: {shipping['express_shipping']['cost']}
- Delivery: {shipping['express_shipping']['delivery_time']}
- Tracking: {shipping['express_shipping']['tracking']}

**Overnight Shipping**:
- Cost: {shipping['overnight_shipping']['cost']}
- Delivery: {shipping['overnight_shipping']['delivery_time']}
- Tracking: {shipping['overnight_shipping']['tracking']}

**International**: {shipping['international_shipping']}

**Premium Members**: Get FREE 2-day shipping on ALL orders!
"""

    elif info_type == "returns":
        returns = company['policies']['returns']
        return f"""**TechGear Return Policy**:

- **Return Window**: {returns['return_window']}
- **Condition**: {returns['condition']}
- **Restocking Fee**: {returns['restocking_fee']}
- **How to Return**: {returns['how_to_return']}

**Exceptions**: {returns['exceptions']}

Premium members enjoy an extended 45-day return window!
"""

    elif info_type == "warranty":
        warranty = company['policies']['warranty']
        return f"""**TechGear Warranty Information**:

- **Standard Warranty**: {warranty['standard_warranty']}
- **Extended Warranty**: {warranty['extended_warranty']}
- **Coverage**: {warranty['coverage']}
- **Claim Process**: {warranty['claim_process']}

All TechGear products include our comprehensive warranty for peace of mind!
"""

    elif info_type == "payment":
        payment = company['payment_methods']
        return f"""**TechGear Payment Methods**:

**Accepted Payment Methods**:
{chr(10).join('- ' + method for method in payment['accepted'])}

**Security**: {payment['security']}

**Payment Plans**: {payment['payment_plans']}

Your payment information is always safe with TechGear!
"""

    else:
        return "Please specify info type: general, contact, shipping, returns, warranty, or payment"


@tool
def search_faqs(question: str) -> str:
    """
    Search TechGear's comprehensive FAQ database to answer customer questions.
    
    This tool searches through ALL company policies, procedures, and information including:
    - Contact information and support channels
    - Payment methods and billing issues  
    - Sign-up and account management
    - Complaints and how to file them
    - Compensation and refund procedures
    - Product feedback submission
    - How to purchase products
    - Account deletion and privacy
    
    Use this tool for ANY customer service question - it has comprehensive coverage!
    
    Args:
        question: Customer's question (e.g., "toll-free number", "payment issue", "complaint", "sign up error")
    
    Returns:
        Detailed answer from FAQ database
    """
    faqs = load_company_faqs()
    
    question_lower = question.lower()
    matches = []
    
    # Search through all FAQs
    for faq in faqs:
        score = 0
        faq_q = faq['question'].lower()
        faq_a = faq['answer'].lower()
        faq_category = faq.get('category', '').lower()
        
        # Match question keywords
        for word in question_lower.split():
            if len(word) > 3:  # Skip short words
                if word in faq_q:
                    score += 10
                if word in faq_a:
                    score += 3
                if word in faq_category:
                    score += 5
        
        # Boost scores for exact intent matches
        intent_keywords = {
            'contact': ['contact', 'phone', 'email', 'support', 'call', 'reach', 'toll-free', 'number'],
            'payment': ['payment', 'pay', 'billing', 'charge', 'card', 'paypal'],
            'complaint': ['complaint', 'complain', 'issue', 'problem', 'report', 'notify'],
            'signup': ['sign', 'register', 'account', 'create', 'error', 'login'],
            'refund': ['refund', 'return', 'money', 'reimburs', 'compensation'],
            'purchase': ['buy', 'purchase', 'order', 'shop', 'get'],
            'feedback': ['feedback', 'review', 'comment', 'suggest'],
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in question_lower for kw in keywords):
                if intent in faq.get('intent', '').lower() or intent in faq_category:
                    score += 15
        
        if score > 0:
            matches.append((score, faq))
    
    # Sort by relevance
    matches.sort(key=lambda x: x[0], reverse=True)
    
    if not matches:
        # Fallback - return contact info
        return """I'd be happy to help! While I couldn't find a specific FAQ for that, you can always reach our support team:

📞 **Phone**: 1-800-TECHGEAR (1-800-832-4432)
📧 **Email**: support@techgear.com  
💬 **Live Chat**: Available 24/7 on our website
🌐 **Website**: www.techgear.com

Our team is ready to assist you!"""
    
    # Return top match(es)
    if len(matches) == 1 or matches[0][0] > matches[1][0] * 1.5:
        # Clear winner
        faq = matches[0][1]
        return f"{faq['answer']}"
    else:
        # Multiple good matches - return top 2
        result = "Here's what I found:\n\n"
        for i, (score, faq) in enumerate(matches[:2], 1):
            result += f"**{i}. {faq['question']}**\n{faq['answer']}\n\n"
        return result


# List of all product tools for easy import
PRODUCT_TOOLS = [
    search_products,
    get_product_details,
    check_product_availability,
    get_product_categories,
    compare_products,
    get_company_info,
    search_faqs  # NEW: Comprehensive FAQ search
]
