import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailService:
    """Handle email operations for OTP delivery"""
    
    def __init__(self, smtp_server='smtp.gmail.com', smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        # You need to set these in your environment or config
        self.sender_email = None
        self.sender_password = None
    
    def configure(self, sender_email, sender_password):
        """Configure email credentials"""
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def send_otp_email(self, recipient_email, otp_code):
        """Send OTP via email"""
        if not self.sender_email or not self.sender_password:
            return False, "Email service not configured. Please add your email credentials."
        
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['From'] = self.sender_email
            message['To'] = recipient_email
            message['Subject'] = 'Your OTP Code - Resume-JD Matcher'
            
            # HTML email body
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f4;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h2 style="color: #1f77b4; text-align: center;">Resume-JD Matcher</h2>
                        <p style="font-size: 16px; color: #333;">Hello,</p>
                        <p style="font-size: 16px; color: #333;">Your OTP code for login verification is:</p>
                        <div style="background-color: #f0f7ff; padding: 20px; text-align: center; margin: 20px 0; border-radius: 5px;">
                            <h1 style="color: #1f77b4; margin: 0; letter-spacing: 5px;">{otp_code}</h1>
                        </div>
                        <p style="font-size: 14px; color: #666;">This OTP will expire in 10 minutes.</p>
                        <p style="font-size: 14px; color: #666;">If you didn't request this code, please ignore this email.</p>
                        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                        <p style="font-size: 12px; color: #999; text-align: center;">AI Resume-JD Matcher | Powered by NLP & AI</p>
                    </div>
                </body>
            </html>
            """
            
            # Attach HTML part
            html_part = MIMEText(html_body, 'html')
            message.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            return True, "OTP sent successfully!"
        
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"