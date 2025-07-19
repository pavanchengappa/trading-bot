# notifications/notifier.py - Notification system
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class Notifier:
    """Notification system for trading bot alerts"""
    
    def __init__(self, config: dict):
        self.config = config
        self.email_enabled = config.get('email_enabled', False)
        self.telegram_enabled = config.get('telegram_enabled', False)
        
        # Email configuration
        self.email_address = config.get('email_address', config.get('email', ''))
        self.smtp_server = "smtp.gmail.com"  # Default Gmail SMTP
        self.smtp_port = 587
        
        # Telegram configuration
        self.telegram_bot_token = config.get('telegram_bot_token', '')
        self.telegram_chat_id = config.get('telegram_chat_id', '')
    
    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """Send notification through all enabled channels"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {title}\n\n{message}"
        
        # Send email notification
        if self.email_enabled:
            self._send_email(title, full_message, priority)
        
        # Send Telegram notification
        if self.telegram_enabled:
            self._send_telegram(title, message)
        
        logger.info(f"Notification sent: {title}")
    
    def _send_email(self, subject: str, message: str, priority: str = "normal"):
        """Send email notification"""
        try:
            if not self.email_address:
                logger.warning("Email address not configured")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = self.email_address
            msg['Subject'] = f"Trading Bot Alert: {subject}"
            
            # Add priority header
            if priority == "high":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            # Note: In a real implementation, you'd need to configure SMTP credentials
            # For now, this is a placeholder that logs the email
            logger.info(f"Email notification (not sent - SMTP not configured):\nSubject: {subject}\nMessage: {message}")
            
            # Example SMTP code (commented out):
            # with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            #     server.starttls()
            #     server.login(self.email_address, "your_app_password")
            #     server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_telegram(self, title: str, message: str):
        """Send Telegram notification"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                logger.warning("Telegram bot token or chat ID not configured")
                return
            
            # Format message for Telegram
            telegram_message = f"🤖 *{title}*\n\n{message}"
            
            # Send message via Telegram Bot API
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': telegram_message,
                'parse_mode': 'MarkdownV2'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
            else:
                logger.error(f"Telegram notification failed: {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def send_trade_notification(self, symbol: str, action: str, quantity: float, price: float):
        """Send trade execution notification"""
        title = f"Trade Executed - {action}"
        message = f"Symbol: {symbol}\nAction: {action}\nQuantity: {quantity:.6f}\nPrice: {price:.2f}"
        self.send_notification(title, message, priority="normal")
    
    def send_error_notification(self, error_type: str, error_message: str):
        """Send error notification"""
        title = f"Trading Bot Error - {error_type}"
        message = f"Error: {error_message}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_notification(title, message, priority="high")
    
    def send_daily_summary(self, summary_data: dict):
        """Send daily trading summary"""
        title = "Daily Trading Summary"
        message = f"""
📊 Daily Trading Summary

Total Trades: {summary_data.get('total_trades', 0)}
Winning Trades: {summary_data.get('winning_trades', 0)}
Losing Trades: {summary_data.get('losing_trades', 0)}
Total P&L: {summary_data.get('total_pnl', 0.0):.2f}
Win Rate: {summary_data.get('win_rate', 0.0):.1%}
Average Win: {summary_data.get('avg_win', 0.0):.2f}
Average Loss: {summary_data.get('avg_loss', 0.0):.2f}
        """.strip()
        
        self.send_notification(title, message, priority="normal")
    
    def send_risk_alert(self, alert_type: str, details: str):
        """Send risk management alert"""
        title = f"Risk Alert - {alert_type}"
        message = f"Risk management triggered:\n{details}"
        self.send_notification(title, message, priority="high")
    
    def test_notifications(self):
        """Test notification channels"""
        logger.info("Testing notification channels...")
        
        # Test email
        if self.email_enabled:
            self._send_email("Test Notification", "This is a test notification from the trading bot.")
        
        # Test Telegram
        if self.telegram_enabled:
            self._send_telegram("Test Notification", "This is a test notification from the trading bot.")
        
        logger.info("Notification test completed")
    
    def get_notification_status(self) -> dict:
        """Get notification system status"""
        return {
            'email_enabled': self.email_enabled,
            'email_address': self.email_address if self.email_enabled else None,
            'telegram_enabled': self.telegram_enabled,
            'telegram_configured': bool(self.telegram_bot_token and self.telegram_chat_id)
        }
