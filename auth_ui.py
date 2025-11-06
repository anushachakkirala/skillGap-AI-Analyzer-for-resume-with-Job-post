import streamlit as st
import re
import os
from PIL import Image

# Custom CSS for authentication pages
# Custom CSS for authentication pages
AUTH_CSS = """
<style>
    body {
        background-color: #0b0d10;
        color: #e0e0e0;
        font-family: 'Poppins', sans-serif;
    }

    /* Auth Container */
    .auth-container {
        max-width: 500px;
        margin: 50px auto;
        padding: 40px;
        background: #1a1c1f;
        border-radius: 10px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.4);
        border: 1px solid #2e2e2e;
    }

    .auth-header {
        text-align: center;
        color: #a78bfa;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 20px;
    }

    .auth-subheader {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 30px;
        font-size: 1rem;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 16px;
        margin-top: 10px;
        box-shadow: 0 0 10px rgba(139,92,246,0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #7C3AED, #8B5CF6);
    }

    .auth-link {
        text-align: center;
        margin-top: 20px;
        color: #9ca3af;
    }

    .auth-link a {
        color: #A78BFA;
        text-decoration: none;
        font-weight: bold;
    }

    .otp-input {
        text-align: center;
        font-size: 20px;
        letter-spacing: 8px;
        font-weight: bold;
        color: #fff;
    }

    /* Profile Page */
    .profile-container {
        background: #1a1c1f;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 15px rgba(0,0,0,0.4);
    }

    .profile-header-section {
        background: transparent;  /* removed white background */
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }

    .profile-avatar-container {
        position: relative;
        width: 150px;
        height: 150px;
        margin: 0 auto 1rem;
    }

    .profile-avatar {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #A78BFA;
        box-shadow: 0 0 15px rgba(139,92,246,0.4);
    }

    .profile-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #E5E7EB;
        margin: 1rem 0 0.5rem;
    }

    .profile-email {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .profile-info-card {
        background: #1f2125;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.4);
        margin-bottom: 1.5rem;
        border: 1px solid #2e2e2e;
    }

    .profile-info-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #A78BFA;
        margin-bottom: 1rem;
    }

    .info-row {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        background: #2a2d31;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }

    .info-label {
        font-weight: 600;
        color: #E5E7EB;
        margin-bottom: 0.2rem;
    }

    .info-value {
        color: #9CA3AF;
        font-size: 0.9rem;
    }

    .edit-mode-card {
        background: #2a2d31;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .upload-area {
        border: 2px dashed #A78BFA;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        background: #1f2125;
        margin: 1rem 0;
        color: #9CA3AF;
    }
</style>
"""


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Valid password"

def show_signup_page(auth_db):
    """Display signup page"""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-header">Create Account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subheader">Join the AI Resume-JD Matcher</div>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            email = st.text_input("Email *", placeholder="your.email@example.com")
            password = st.text_input("Password *", type="password", placeholder="Min 8 characters")
            confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Re-enter password")
            
            st.markdown("---")
            
            
            phone_number = st.text_input("Phone Number", placeholder="+91 0000000000")
            job_role = st.text_input("Job Role", placeholder="e.g., Software Engineer")
            company_name = st.text_input("Company Name", placeholder="Your company")
            location = st.text_input("Location", placeholder="City, Country")
            
            submit = st.form_submit_button("Create Account")
            
            if submit:
                # Validation
                if not name or not email or not password:
                    st.error("Please fill in all required fields (marked with *)")
                elif not validate_email(email):
                    st.error("Please enter a valid email address")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    is_valid, msg = validate_password(password)
                    if not is_valid:
                        st.error(msg)
                    else:
                        # Create user
                        success, message = auth_db.create_user(
                            name, email, password, phone_number, 
                            job_role, company_name, location
                        )
                        
                        if success:
                            st.success(message)
                            st.info("Please go to Login page to sign in")
                            st.session_state.show_login = True
                        else:
                            st.error(message)
        
        st.markdown('<div class="auth-link">Already have an account? <a href="#" id="login-link">Login here</a></div>', 
                   unsafe_allow_html=True)
        
        if st.button("← Back to Login"):
            st.session_state.auth_page = 'login'
            st.rerun()

def show_login_page(auth_db, email_service):
    """Display login page"""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-header"> Welcome Back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subheader">Sign in to continue</div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submit = st.form_submit_button("Login")
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                elif not validate_email(email):
                    st.error("Please enter a valid email address")
                else:
                    # Verify credentials
                    success, user_data = auth_db.verify_user(email, password)
                    
                    if success:
                        # Generate and send OTP
                        otp = auth_db.generate_otp(email)
                        email_success, email_msg = email_service.send_otp_email(email, otp)
                        
                        if email_success:
                            st.session_state.pending_user = user_data
                            st.session_state.pending_email = email
                            st.session_state.auth_page = 'otp'
                            st.success("OTP sent to your email!")
                            st.rerun()
                        else:
                            st.error(email_msg)
                            st.info("Demo Mode: Your OTP is: " + otp)
                            st.session_state.pending_user = user_data
                            st.session_state.pending_email = email
                            st.session_state.demo_otp = otp
                            st.session_state.auth_page = 'otp'
                    else:
                        st.error("Invalid email or password")
        
        st.markdown("---")
        
        if st.button("Create New Account"):
            st.session_state.auth_page = 'signup'
            st.rerun()

def show_otp_page(auth_db):
    """Display OTP verification page"""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-header">🔐 Verify OTP</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="auth-subheader">Enter the code sent to {st.session_state.pending_email}</div>', 
                   unsafe_allow_html=True)
        
        with st.form("otp_form"):
            otp = st.text_input("Enter 6-digit OTP", max_chars=6, placeholder="000000")
            submit = st.form_submit_button("Verify")
            
            if submit:
                if len(otp) != 6 or not otp.isdigit():
                    st.error("Please enter a valid 6-digit OTP")
                else:
                    # Check demo mode first
                    if 'demo_otp' in st.session_state and otp == st.session_state.demo_otp:
                        verified = True
                    else:
                        verified = auth_db.verify_otp(st.session_state.pending_email, otp)
                    
                    if verified:
                        st.session_state.authenticated = True
                        st.session_state.user = st.session_state.pending_user
                        st.session_state.auth_page = 'main'
                        
                        # Clean up
                        if 'demo_otp' in st.session_state:
                            del st.session_state.demo_otp
                        if 'pending_user' in st.session_state:
                            del st.session_state.pending_user
                        if 'pending_email' in st.session_state:
                            del st.session_state.pending_email
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid or expired OTP")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Back to Login"):
                st.session_state.auth_page = 'login'
                st.rerun()
        
        with col_b:
            if st.button("Resend OTP"):
                otp = auth_db.generate_otp(st.session_state.pending_email)
                st.info(f"Demo Mode - New OTP: {otp}")
                st.session_state.demo_otp = otp

def show_profile_page(auth_db):
    """Display user profile page with profile picture and modern design"""
    import os
    from PIL import Image
    import io
    
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    user_details = auth_db.get_user_details(st.session_state.user['email'])
    
    if not user_details:
        st.error("Failed to load user details")
        return
    
    # Edit mode toggle
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
    
    # Profile Header Section
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    
    if not st.session_state.edit_mode:
        # VIEW MODE - Display Profile
        st.markdown('<div class="profile-header-section">', unsafe_allow_html=True)
        
        # Profile Picture
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            profile_pic = user_details.get('profile_pic', '')
            if profile_pic and os.path.exists(profile_pic):
                try:
                    profile_img = Image.open(profile_pic)
                    st.image(profile_img, width=150, use_conta_width=False)
                except:
                    st.markdown("""
                    <div class="profile-avatar-container">
                        <div class="profile-avatar" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; font-size: 4rem; color: white;">
                            👤
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="profile-avatar-container">
                    <div class="profile-avatar" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; font-size: 4rem; color: white;">
                        👤
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="profile-name">{user_details["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-email">✉️ {user_details["email"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Personal Information Card
        st.markdown('<div class="profile-info-card">', unsafe_allow_html=True)
        st.markdown('<div class="profile-info-header">👤 Personal Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-row">
                <div class="info-icon">📱</div>
                <div>
                    <div class="info-label">Phone Number</div>
                    <div class="info-value">{user_details.get('phone_number', '') or 'Not provided'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-row">
                <div class="info-icon">📍</div>
                <div>
                    <div class="info-label">Location</div>
                    <div class="info-value">{user_details.get('location', '') or 'Not provided'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Professional Information Card
        st.markdown('<div class="profile-info-card">', unsafe_allow_html=True)
        st.markdown('<div class="profile-info-header">💼 Professional Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-row">
                <div class="info-icon">💻</div>
                <div>
                    <div class="info-label">Job Role</div>
                    <div class="info-value">{user_details.get('job_role', '') or 'Not provided'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-row">
                <div class="info-icon">🏢</div>
                <div>
                    <div class="info-label">Company</div>
                    <div class="info-value">{user_details.get('company_name', '') or 'Not provided'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✏️ Edit Profile", use_container_width=True, type="primary"):
                st.session_state.edit_mode = True
                st.rerun()
        
        with col2:
            if st.button("🔙 Back to App", use_container_width=True):
                st.session_state.show_profile = False
                st.session_state.edit_mode = False
                st.rerun()
    
    else:
        # EDIT MODE
        st.markdown('<div class="edit-mode-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white; text-align: center;">✏️ Edit Your Profile</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.form("profile_edit_form"):
            # Profile Picture Upload
            st.markdown("### 📸 Profile Picture")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                profile_pic = user_details.get('profile_pic', '')
                if profile_pic and os.path.exists(profile_pic):
                    try:
                        current_img = Image.open(profile_pic)
                        st.image(current_img, caption="Current Picture", width=150)
                    except:
                        st.info("No current profile picture")
                else:
                    st.markdown("""
                    <div style="width: 150px; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 3rem; color: white;">
                        👤
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                uploaded_file = st.file_uploader(
                    "Upload New Profile Picture",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload a square image for best results (PNG, JPG, JPEG)"
                )
                
                if uploaded_file:
                    st.success("✅ New picture selected!")
            
            st.markdown("---")
            
            # Personal Information
            st.markdown("### 👤 Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", value=user_details['name'])
            
            with col2:
                phone_number = st.text_input("Phone Number", value=user_details.get('phone_number', ''), placeholder="+1234567890")
            
            location = st.text_input("Location", value=user_details.get('location', ''), placeholder="City, Country")
            
            st.markdown("---")
            
            # Professional Information
            st.markdown("### 💼 Professional Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                job_role = st.text_input("Job Role", value=user_details.get('job_role', ''), placeholder="e.g., Software Engineer")
            
            with col2:
                company_name = st.text_input("Company Name", value=user_details.get('company_name', ''), placeholder="Your company")
            
            st.markdown("---")
            
            # Submit Buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                submit = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
            
            with col2:
                cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            if submit:
                if not name:
                    st.error("Name cannot be empty")
                else:
                    profile_pic_path = user_details.get('profile_pic', '')
                    
                    # Handle profile picture upload
                    if uploaded_file:
                        # Create unique filename
                        file_extension = uploaded_file.name.split('.')[-1]
                        safe_email = user_details['email'].replace('@', '_at_').replace('.', '_')
                        filename = f"profile_{safe_email}.{file_extension}"
                        filepath = os.path.join('uploads', filename)
                        
                        # Save the uploaded file
                        try:
                            image = Image.open(uploaded_file)
                            # Resize image to 300x300 for consistency
                            image = image.resize((300, 300), Image.Resampling.LANCZOS)
                            image.save(filepath)
                            profile_pic_path = filepath
                            st.success("Profile picture uploaded successfully!")
                        except Exception as e:
                            st.error(f"Failed to upload picture: {str(e)}")
                    
                    # Update profile
                    success, message = auth_db.update_user_profile(
                        user_details['email'], name, phone_number, job_role, 
                        company_name, location, profile_pic_path
                    )
                    
                    if success:
                        st.success(message)
                        st.session_state.user['name'] = name
                        st.session_state.edit_mode = False
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(message)
            
            if cancel:
                st.session_state.edit_mode = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)