# route_frontend_profile.py

from config import *
from functions_authentication import *

def register_route_frontend_profile(app):
    @app.route('/profile')
    @login_required
    def profile():
        user = session.get('user')
        return render_template('profile.html', user=user)