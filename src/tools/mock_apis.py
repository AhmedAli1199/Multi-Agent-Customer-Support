"""
Mock backend APIs for simulating customer support actions.
These are simple Python functions that simulate real backend operations.
"""
import random
from typing import Dict, Optional
from datetime import datetime, timedelta

class MockOrderAPI:
    """Mock order management API"""

    def __init__(self):
        # Simulate some existing orders
        self.orders = {
            "12345": {
                "order_id": "12345",
                "customer_id": "CUST001",
                "status": "shipped",
                "items": ["Laptop", "Mouse"],
                "total": 1299.99,
                "created_date": "2024-11-20",
                "shipped_date": "2024-11-22"
            },
            "67890": {
                "order_id": "67890",
                "customer_id": "CUST002",
                "status": "processing",
                "items": ["Phone Case"],
                "total": 29.99,
                "created_date": "2024-11-28",
                "shipped_date": None
            }
        }

    def check_order_status(self, order_id: str) -> Dict:
        """Check the status of an order"""
        if order_id in self.orders:
            return {
                "success": True,
                "order": self.orders[order_id]
            }
        else:
            return {
                "success": False,
                "error": f"Order {order_id} not found"
            }

    def cancel_order(self, order_id: str, reason: str) -> Dict:
        """Cancel an order"""
        if order_id not in self.orders:
            return {
                "success": False,
                "error": f"Order {order_id} not found"
            }

        order = self.orders[order_id]

        if order["status"] == "delivered":
            return {
                "success": False,
                "error": "Cannot cancel delivered order. Please request a return instead."
            }

        # Simulate cancellation
        order["status"] = "cancelled"
        order["cancellation_reason"] = reason
        order["cancelled_date"] = datetime.now().strftime("%Y-%m-%d")

        return {
            "success": True,
            "message": f"Order {order_id} has been cancelled",
            "refund_amount": order["total"]
        }

    def modify_order(self, order_id: str, changes: Dict) -> Dict:
        """Modify an order (e.g., change shipping address)"""
        if order_id not in self.orders:
            return {
                "success": False,
                "error": f"Order {order_id} not found"
            }

        order = self.orders[order_id]

        if order["status"] not in ["processing", "pending"]:
            return {
                "success": False,
                "error": f"Cannot modify order with status: {order['status']}"
            }

        # Apply changes
        order.update(changes)
        order["modified_date"] = datetime.now().strftime("%Y-%m-%d")

        return {
            "success": True,
            "message": f"Order {order_id} has been modified",
            "order": order
        }

class MockRefundAPI:
    """Mock refund processing API"""

    def __init__(self):
        self.refunds = {}

    def initiate_refund(self, order_id: str, amount: float, reason: str) -> Dict:
        """Initiate a refund"""
        refund_id = f"REF{random.randint(10000, 99999)}"

        refund = {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "status": "pending",
            "initiated_date": datetime.now().strftime("%Y-%m-%d"),
            "estimated_completion": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        }

        self.refunds[refund_id] = refund

        return {
            "success": True,
            "refund_id": refund_id,
            "message": f"Refund of ${amount:.2f} initiated. Expected in 5-7 business days.",
            "refund": refund
        }

    def check_refund_eligibility(self, order_id: str) -> Dict:
        """Check if an order is eligible for refund"""
        # Simulate eligibility check
        # In reality, this would check order date, return policy, etc.
        eligible = random.choice([True, True, True, False])  # 75% eligible

        return {
            "success": True,
            "eligible": eligible,
            "reason": "Within 30-day return window" if eligible else "Outside return window"
        }

    def check_refund_status(self, refund_id: str) -> Dict:
        """Check status of a refund"""
        if refund_id in self.refunds:
            return {
                "success": True,
                "refund": self.refunds[refund_id]
            }
        else:
            return {
                "success": False,
                "error": f"Refund {refund_id} not found"
            }

class MockAccountAPI:
    """Mock account management API"""

    def __init__(self):
        self.accounts = {
            "CUST001": {
                "customer_id": "CUST001",
                "email": "customer1@example.com",
                "name": "John Doe",
                "phone": "+1-555-0001",
                "address": "123 Main St, City, State 12345"
            }
        }

    def update_address(self, customer_id: str, new_address: str) -> Dict:
        """Update customer shipping address"""
        if customer_id not in self.accounts:
            return {
                "success": False,
                "error": f"Customer {customer_id} not found"
            }

        self.accounts[customer_id]["address"] = new_address
        self.accounts[customer_id]["address_updated"] = datetime.now().strftime("%Y-%m-%d")

        return {
            "success": True,
            "message": "Address updated successfully",
            "new_address": new_address
        }

    def reset_password(self, customer_id: str) -> Dict:
        """Send password reset link"""
        if customer_id not in self.accounts:
            return {
                "success": False,
                "error": f"Customer {customer_id} not found"
            }

        email = self.accounts[customer_id]["email"]

        return {
            "success": True,
            "message": f"Password reset link sent to {email}"
        }

    def get_account_info(self, customer_id: str) -> Dict:
        """Get account information"""
        if customer_id in self.accounts:
            return {
                "success": True,
                "account": self.accounts[customer_id]
            }
        else:
            return {
                "success": False,
                "error": f"Customer {customer_id} not found"
            }

# Global instances
order_api = MockOrderAPI()
refund_api = MockRefundAPI()
account_api = MockAccountAPI()
