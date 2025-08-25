# route_frontend_groups.py

from config import *
from functions_authentication import *
from functions_settings import *

def register_route_frontend_groups(app):
    @app.route("/my_groups", methods=["GET"])
    @login_required
    @user_required
    @enabled_required("enable_group_workspaces")
    def my_groups():
        """
        Renders the My Groups page (templates/my_groups.html).
        """
        
        return render_template("my_groups.html")

    @app.route("/groups/<group_id>", methods=["GET"])
    @login_required
    @user_required
    @enabled_required("enable_group_workspaces")
    def manage_group(group_id):
        """
        Renders a page or view for managing a single group (not shown in detail here).
        Could be a second template like 'manage_group.html'.
        """
        
        return render_template("manage_group.html", group_id=group_id)
