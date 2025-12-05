"""
LangChain tools for Action Agent - proper tool calling instead of JSON planning.
These tools wrap the mock APIs and make them callable by the LLM.
"""
from typing import Optional, Dict
from langchain.tools import tool
from tools.mock_apis import order_api, refund_api, account_api
import json


@tool
def check_order_status(order_id: str) -> str:
    """
    Check the current status of a customer order.

    Args:
        order_id: The order ID to check (e.g., "12345")

    Returns:
        String with order status information
    """
    result = order_api.check_order_status(order_id)

    if result["success"]:
        order = result["order"]
        return f"""Order {order_id} Status:
- Status: {order['status']}
- Items: {', '.join(order['items'])}
- Total: ${order['total']}
- Order Date: {order['created_date']}
- Shipped Date: {order.get('shipped_date', 'Not yet shipped')}
"""
    else:
        return f"ORDER_NOT_FOUND: I couldn't find an order with ID '{order_id}'. Please ask the customer to double-check their order ID. They can find it in their order confirmation email or their account order history."


@tool
def cancel_order(order_id: str, reason: str = "Customer request") -> str:
    """
    Cancel a customer order. Can only cancel orders that haven't been delivered.

    Args:
        order_id: The order ID to cancel (e.g., "12345")
        reason: Reason for cancellation (optional)

    Returns:
        Confirmation message or error
    """
    result = order_api.cancel_order(order_id, reason)

    if result["success"]:
        return f"""Order {order_id} has been successfully cancelled.
- Refund Amount: ${result['refund_amount']}
- Reason: {reason}
- Refund will be processed in 5-7 business days to your original payment method.

Is there anything else I can help you with?"""
    else:
        error = result['error']
        if "not found" in error.lower():
            return f"ORDER_NOT_FOUND: I couldn't find an order with ID '{order_id}'. Please ask the customer to verify their order ID from their confirmation email."
        else:
            return f"CANNOT_CANCEL: {error}. Please inform the customer about this limitation."


@tool
def modify_order(order_id: str, new_address: Optional[str] = None, shipping_upgrade: Optional[str] = None) -> str:
    """
    Modify an existing order (change shipping address or upgrade shipping).
    Can only modify orders in 'processing' or 'pending' status.

    Args:
        order_id: The order ID to modify
        new_address: New shipping address (if changing address)
        shipping_upgrade: New shipping method (e.g., "express", "overnight")

    Returns:
        Confirmation message or error
    """
    changes = {}
    if new_address:
        changes["shipping_address"] = new_address
    if shipping_upgrade:
        changes["shipping_method"] = shipping_upgrade

    result = order_api.modify_order(order_id, changes)

    if result["success"]:
        return f"""Order {order_id} has been successfully modified.
Changes applied: {', '.join(f'{k}: {v}' for k, v in changes.items())}

Updated order details available in your account."""
    else:
        return f"Unable to modify order: {result['error']}"


@tool
def initiate_refund(order_id: str, amount: float, reason: str = "Customer request") -> str:
    """
    Initiate a refund for a customer order.

    Args:
        order_id: The order ID to refund
        amount: Refund amount in USD
        reason: Reason for refund

    Returns:
        Refund confirmation with refund ID and timeline
    """
    result = refund_api.initiate_refund(order_id, amount, reason)

    if result["success"]:
        refund = result["refund"]
        return f"""Refund initiated successfully!

- Refund ID: {refund['refund_id']}
- Order ID: {order_id}
- Amount: ${amount:.2f}
- Status: {refund['status']}
- Initiated: {refund['initiated_date']}
- Expected Completion: {refund['estimated_completion']}

You'll receive an email confirmation shortly. The refund will appear in your account within 5-7 business days."""
    else:
        return f"Unable to process refund: {result['error']}"


@tool
def check_refund_status(refund_id: str) -> str:
    """
    Check the status of an existing refund.

    Args:
        refund_id: The refund ID to check (e.g., "REF12345")

    Returns:
        Current refund status
    """
    result = refund_api.check_refund_status(refund_id)

    if result["success"]:
        refund = result["refund"]
        return f"""Refund {refund_id} Status:
- Order ID: {refund['order_id']}
- Amount: ${refund['amount']}
- Status: {refund['status']}
- Initiated: {refund['initiated_date']}
- Expected Completion: {refund['estimated_completion']}
"""
    else:
        return f"Refund not found: {result['error']}"


@tool
def update_customer_address(customer_id: str, new_address: str) -> str:
    """
    Update a customer's default shipping address.

    Args:
        customer_id: Customer ID (e.g., "CUST001")
        new_address: New shipping address

    Returns:
        Confirmation message
    """
    result = account_api.update_address(customer_id, new_address)

    if result["success"]:
        return f"""Shipping address updated successfully!

New address: {result['new_address']}

This address will be used for all future orders."""
    else:
        return f"Unable to update address: {result['error']}"


@tool
def reset_customer_password(customer_id: str) -> str:
    """
    Send a password reset link to customer's email.

    Args:
        customer_id: Customer ID

    Returns:
        Confirmation that reset link was sent
    """
    result = account_api.reset_password(customer_id)

    if result["success"]:
        return result["message"]
    else:
        return f"Unable to reset password: {result['error']}"


@tool
def get_customer_account_info(customer_id: str) -> str:
    """
    Retrieve customer account information.

    Args:
        customer_id: Customer ID

    Returns:
        Customer account details
    """
    result = account_api.get_account_info(customer_id)

    if result["success"]:
        account = result["account"]
        return f"""Account Information:
- Customer ID: {account['customer_id']}
- Name: {account['name']}
- Email: {account['email']}
- Phone: {account['phone']}
- Address: {account['address']}
"""
    else:
        return f"Unable to retrieve account: {result['error']}"


# List of all action tools for easy import
ACTION_TOOLS = [
    check_order_status,
    cancel_order,
    modify_order,
    initiate_refund,
    check_refund_status,
    update_customer_address,
    reset_customer_password,
    get_customer_account_info
]
