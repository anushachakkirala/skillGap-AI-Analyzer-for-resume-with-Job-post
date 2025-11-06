"""
Setup script to create required folders and initialize the application
Run this once before starting the app
"""

import os

def setup_application():
    """Create necessary folders for the application"""
    
    folders = ['uploads']
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created folder: {folder}")
        else:
            print(f"ℹ️  Folder already exists: {folder}")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Database
users.db
*.db

# Uploads folder (profile pictures)
uploads/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Streamlit
.streamlit/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")
    else:
        print("ℹ️  .gitignore already exists")
    
    print("\n✨ Setup complete! You can now run: streamlit run app.py")

if __name__ == "__main__":
    setup_application()