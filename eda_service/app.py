import random, datetime, psycopg2, smtplib
from dotenv import load_dotenv
from email.message import EmailMessage
import requests
from report_generator import generate_eda_report
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory,send_file
from werkzeug.utils import secure_filename
from pycaret_service import run_automl_pipeline
import os
import pandas as pd


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

DB_CONFIG = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT")
}

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/models", exist_ok=True)

 # Load environment variables from .env file

# ---------------- UTILS ----------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_dataset(file_path):
    """Smart file reader that detects actual format"""
    # Try excel first if extension says so
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        try:
            return pd.read_excel(file_path)
        except Exception:
            pass
    
    # Try CSV with different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    
    # Last resort
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def generate_username(email):
    prefix = email.split('@')[0][:5]
    now = datetime.datetime.now()
    return f"{prefix}_{now.strftime('%d%m')}"


def send_otp(receiver_email):
    otp = str(random.randint(100000, 999999))

    msg = EmailMessage()
    msg.set_content(f"Your OTP code is: {otp}")
    msg['Subject'] = 'DataPulse OTP Verification'
    msg['From'] = os.getenv("GMAIL_USER")
    msg['To'] = receiver_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.getenv("GMAIL_USER"), os.getenv("GMAIL_APP_PASSWORD"))
        smtp.send_message(msg)

    return otp


# ---------------- HOME ----------------

@app.route('/')
def home():
    return render_template('login.html')


# ---------------- AUTH ----------------

@app.route('/auth', methods=['POST'])
def auth():
    mode = request.form.get('mode')
    password = request.form.get('password')

    conn = get_db_connection()
    cur = conn.cursor()

    # -------- SIGNUP --------
    if mode == 'signup':
        email = request.form.get('email')

        session['temp_user'] = {
            'email': email,
            'password': password,
            'otp': send_otp(email)
        }

        flash("OTP sent to your email", "success")
        cur.close()
        conn.close()
        return render_template('verify.html',mode='signup')

    # -------- LOGIN --------
    else:
        username = request.form.get('username')

        # ✅ If user already failed once → now send OTP
        if session.get('otp_candidate'):
            data = session.pop('otp_candidate')
            otp = send_otp(data['email'])

            session['login_otp'] = otp
            session['login_user'] = data['username']

            flash("OTP sent to your email.", "success")
            cur.close()
            conn.close()
            return redirect(url_for('verify_login_page'))

        # Check username + password
        cur.execute(
            "SELECT email FROM users WHERE username = %s AND password = %s",
            (username, password)
        )
        user = cur.fetchone()

        if user:
            session['user'] = username
            cur.close()
            conn.close()
            return redirect(url_for('upload_page'))

        # Check if username exists
        cur.execute(
            "SELECT email FROM users WHERE username = %s",
            (username,)
        )
        user_email = cur.fetchone()

        if user_email:
            session['otp_candidate'] = {
                'username': username,
                'email': user_email[0]
            }

            flash("Incorrect password. Click login again to receive OTP.", "error")
            cur.close()
            conn.close()
            return redirect(url_for('home'))

        flash("User not found", "error")
        cur.close()
        conn.close()
        return redirect(url_for('home'))


# ---------------- OTP VERIFY ----------------

# GET → Show OTP page
@app.route('/verify-login-otp', methods=['GET'])
def verify_login_page():
    return render_template('verify.html',mode='login')


# POST → Process OTP
@app.route('/verify-login-otp', methods=['POST'])
def verify_login_otp():
    entered_otp = request.form.get('otp')
    session_otp = session.get('login_otp')
    username = session.get('login_user')

    if entered_otp == session_otp:
        session['user'] = username
        session.pop('login_otp', None)
        session.pop('login_user', None)

        
        return redirect(url_for('upload_page'))

    flash("Invalid OTP", "error")
    return render_template('verify.html')


# -------- SIGNUP OTP VERIFY --------
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    user_otp = request.form.get('otp')
    temp = session.get('temp_user')

    if temp and user_otp == temp['otp']:
        username = generate_username(temp['email'])

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, temp['email'], temp['password'])
        )

        conn.commit()
        cur.close()
        conn.close()

        session.pop('temp_user', None)
        session['user'] = username

        return redirect(url_for('upload_page'))

    flash("Invalid OTP", "error")
    return render_template('verify.html')


# ---------------- UPLOAD PAGE ----------------

@app.route('/upload')
def upload_page():
    username = session.get('user')
    if not username:
        flash("Please login first", "error")
        return redirect(url_for('home'))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT file_name, file_path, status, uploaded_at
        FROM datasets
        WHERE user_id = (SELECT user_id FROM users WHERE username=%s)
        ORDER BY uploaded_at DESC
        LIMIT 10
    """, (username,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    recent_datasets = []
    for row in rows:
        recent_datasets.append({
            'file_name': row[0],
            'file_path': row[1],
            'status': row[2],
            'uploaded_at': row[3].strftime("%d %b %Y %H:%M")
        })

    return render_template('upload.html', recent_datasets=recent_datasets)


# ---------------- FILE UPLOAD ----------------

@app.route("/upload", methods=["POST"])
def upload_file():
    if "dataset" not in request.files:
        flash("No file selected", "error")
        return redirect(url_for("upload_page"))

    file = request.files["dataset"]

    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("upload_page"))

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        file_path = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        file.save(file_path)

        try:
            report_filename, complexity_score, complexity_category, complexity_breakdown = generate_eda_report(file_path)
            session["complexity_score"] = complexity_score
            session["complexity_category"] = complexity_category
            session["complexity_breakdown"] = complexity_breakdown

            # ✅ INSERT INTO DATASETS TABLE 
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO datasets (user_id, file_name, file_path, status, uploaded_at)
                VALUES (
                    (SELECT user_id FROM users WHERE username=%s),
                    %s,
                    %s,
                    %s,
                    NOW()
                )
            """, (session['user'], filename, file_path, "Completed"))

            conn.commit()
            cur.close()
            conn.close()

            

            session["current_dataset"] = file_path
            session["current_report"] = report_filename

            return redirect(url_for("report_dashboard"))

        except Exception as e:
            flash(f"Error generating report: {str(e)}", "error")
            return redirect(url_for("upload_page"))

    flash("Invalid file type", "error")
    return redirect(url_for("upload_page"))


# ---------------- REPORT ----------------

@app.route("/report/<filename>")
def view_report(filename):
    return send_from_directory("static/reports", filename)


@app.route("/report-dashboard")
def report_dashboard():
    report_filename = session.get("current_report")
    dataset_path = session.get("current_dataset")

    return render_template(
        "report_view.html",
        report_file=report_filename,
        dataset_loaded=True if dataset_path else False,
        complexity_score=session.get("complexity_score", 0),
        complexity_category=session.get("complexity_category", "Simple"),
        complexity_breakdown=session.get("complexity_breakdown", {}),
    )


# ---------------- CHAT ----------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."})

    report_filename = session.get("current_report")
    if not report_filename:
        return jsonify({"answer": "No report loaded."})

    report_path = os.path.abspath(os.path.join("static", "reports", report_filename))

    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"report_path": report_path, "question": question}
        )
        answer = response.json().get("answer", "No answer returned.")
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"Error calling chat API: {str(e)}"})




@app.route("/get-columns")
def get_columns():
    file_path = session.get("current_dataset")

    if not file_path or not os.path.exists(file_path):
        return jsonify({"columns": []})

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"columns": []})

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        return jsonify({"columns": numeric_cols})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"columns": []})


@app.route("/get-3d-data", methods=["POST"])
def get_3d_data():
    data = request.get_json()
    col_x = data.get("x")
    col_y = data.get("y")
    col_z = data.get("z")

    file_path = session.get("current_dataset")

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No dataset loaded"})

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df = df[[col_x, col_y, col_z]].dropna().head(5000)

        return jsonify({
            "x": df[col_x].tolist(),
            "y": df[col_y].tolist(),
            "z": df[col_z].tolist()
        })

    except Exception as e:
        print("3D ERROR:", str(e))
        return jsonify({"error": str(e)})



@app.route("/prediction_engine")
def prediction_engine():
    dataset_path = session.get("current_dataset")
    print("SESSION DATASET:", session.get("current_dataset"))

    if dataset_path is None:
        flash("Please upload dataset first.")
        return redirect(url_for("upload"))

    df=read_dataset(dataset_path)

    columns = df.columns

    return render_template(
        "prediction_engine.html",
        columns=columns
    )

@app.route("/run_automl", methods=["POST"])
def run_automl():

    target = request.form["target"]
    dataset_path = session.get("current_dataset")

    if not dataset_path:
        flash("Dataset not found.", "error")
        return redirect(url_for("prediction_engine"))
    
    # Get parameters from form
    train_size = float(request.form.get("train_size", 0.8))
    normalize = request.form.get("normalize", "false") == "true"
    feature_selection = request.form.get("feature_selection", "false") == "true"
    remove_outliers = request.form.get("remove_outliers", "false") == "true"
    tune_model = request.form.get("tune_model", "false") == "true"
    
    # Get prediction type and time series parameters
    prediction_type = request.form.get("prediction_type", "normal")
    date_column = request.form.get("date_column", None)
    forecast_periods = int(request.form.get("forecast_periods", 30))

    # Run enhanced AutoML pipeline
    (results, model, problem_type, feature_importance, 
     confusion_plot, actual_vs_pred_plot, model_filename,distribution_plot) = run_automl_pipeline(
        dataset_path, 
        target,
        train_size=train_size,
        normalize=normalize,
        feature_selection=feature_selection,
        remove_outliers=remove_outliers,
        tune_best_model=tune_model,
        prediction_type=prediction_type,
        date_column=date_column,
        forecast_periods=forecast_periods
    )
    
    # Store model filename in session for download
    session['model_filename'] = model_filename

    # Only create model documentation links for Classification/Regression (NOT Time Series)
# Only create model documentation links for Classification/Regression (NOT Time Series)
    if problem_type == "Classification":
        model_docs = {
            "Logistic Regression":"https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
            "Ridge Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html",
            "Extra Trees Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
            "Naive Bayes":"https://scikit-learn.org/stable/modules/naive_bayes.html",
            "K Neighbors Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
            "Linear Discriminant Analysis":"https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html",
            "Decision Tree Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
            "Random Forest Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
            "Ada Boost Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
            "Gradient Boosting Classifier":"https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
            "SVM - Linear Kernel":"https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
        }
        
        # Make model names clickable - ONLY FOR CLASSIFICATION
        results["Model"] = results["Model"].apply(
            lambda x: f'<a href="{model_docs.get(x,"#")}" target="_blank" class="text-blue-500 underline">{x}</a>'
        )
        
    elif problem_type == "Regression":
        model_docs = {
            "Ridge Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
            "Linear Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
            "Lasso Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
            "Least Angle Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html",
            "Lasso Least Angle Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html",
            "Light Gradient Boosting Machine": "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html",
            "Random Forest Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
            "Elastic Net": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
            "Extra Trees Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
            "Gradient Boosting Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
            "AdaBoost Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
            "Orthogonal Matching Pursuit": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html",
            "Bayesian Ridge": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html",
            "K Neighbors Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
            "Huber Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html",
            "Decision Tree Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
            "Dummy Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html",
            "Passive Aggressive Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html",
        }
        
        # Make model names clickable - ONLY FOR REGRESSION
        results["Model"] = results["Model"].apply(
            lambda x: f'<a href="{model_docs.get(x,"#")}" target="_blank" class="text-blue-500 underline">{x}</a>'
        )

    # ❌ DELETE THIS SECTION IF IT EXISTS (around line 541-543):
    # results["Model"] = results["Model"].apply(
    #     lambda x: f'<a href="{model_docs.get(x,"#")}" target="_blank" class="text-blue-500 underline">{x}</a>'
    # )

    # Best model info (only for Classification/Regression)
    if problem_type in ["Classification", "Regression"]:
        best_model_name = results.iloc[0]["Model"]
        
        if problem_type == "Classification":
            best_score = round(results.iloc[0]["Accuracy"], 3)
        else:
            best_score = round(results.iloc[0]["R2"], 3)
        
        # Convert results for charts
        metrics_data = results.to_dict(orient="records")
        tables_html = results.to_html(classes="table-auto w-full text-center", escape=False)
    else:
        # Time Series
        best_model_name = "Prophet Time Series Model"
        best_score = None
        metrics_data = None
        if results is not None:
            # Rename columns for better display
            forecast_display = results.copy()
            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            tables_html = forecast_display.to_html(classes="table-auto w-full text-center", escape=False, index=False)
        else:
            tables_html = None


    # Convert results to HTML table
    return render_template(
        "prediction_engine.html",
        columns=read_dataset(dataset_path).columns,
        tables=tables_html,
        model=str(model),
        problem_type=problem_type,
        best_model=best_model_name,
        best_score=best_score,
        metrics_data=metrics_data,
        feature_importance=feature_importance,
        confusion_matrix_plot=confusion_plot,
        actual_vs_pred_plot=actual_vs_pred_plot,
        distribution_plot=distribution_plot,
        model_filename=model_filename
    )






# Add this new route for model download:

@app.route("/download-model/<filename>")
def download_model(filename):
    """
    Download the trained model file
    """
    try:
        model_path = os.path.join("static", "models", filename)
        
        if not os.path.exists(model_path):
            flash("Model file not found", "error")
            return redirect(url_for("prediction_engine"))
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        flash(f"Error downloading model: {str(e)}", "error")
        return redirect(url_for("prediction_engine"))



@app.route("/generate-story", methods=["POST"])
def generate_story():
    dataset_path = session.get("current_dataset")

    if not dataset_path:
        return jsonify({"error": "No dataset loaded"})

    try:
        response = requests.post(
            "http://127.0.0.1:8000/story",
            json={"csv_path": dataset_path}
        )
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": f"Story service unavailable: {str(e)}"})



@app.route("/admin")
def admin_dashboard():
    if session.get('user') != 'admin':
        flash("Access denied.", "error")
        return redirect(url_for('home'))

    conn = get_db_connection()
    cur = conn.cursor()

    # Total users
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    # Total datasets
    cur.execute("SELECT COUNT(*) FROM datasets")
    total_datasets = cur.fetchone()[0]

    # Total completed
    cur.execute("SELECT COUNT(*) FROM datasets WHERE status = 'Completed'")
    total_completed = cur.fetchone()[0]

    # New users this month
    cur.execute("""
        SELECT COUNT(*) FROM users 
        WHERE DATE_TRUNC('month', created_at) = DATE_TRUNC('month', CURRENT_DATE)
    """)
    new_users_month = cur.fetchone()[0]

    # All users with dataset count
    cur.execute("""
        SELECT u.user_id, u.username, u.email, u.created_at,
               COUNT(d.dataset_id) as dataset_count
        FROM users u
        LEFT JOIN datasets d ON u.user_id = d.user_id
        GROUP BY u.user_id, u.username, u.email, u.created_at
        ORDER BY u.created_at DESC
    """)
    users = cur.fetchall()

    # Recent 20 uploads across all users
    cur.execute("""
        SELECT d.dataset_id, d.file_name, d.status, d.uploaded_at,
               u.username
        FROM datasets d
        JOIN users u ON d.user_id = u.user_id
        ORDER BY d.uploaded_at DESC
        LIMIT 20
    """)
    recent_uploads = cur.fetchall()

    # Uploads per day last 7 days
    cur.execute("""
        SELECT DATE(uploaded_at) as day, COUNT(*) as count
        FROM datasets
        WHERE uploaded_at >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(uploaded_at)
        ORDER BY day
    """)
    uploads_chart = cur.fetchall()

    # Most active users (top 5)
    cur.execute("""
        SELECT u.username, COUNT(d.dataset_id) as uploads
        FROM users u
        JOIN datasets d ON u.user_id = d.user_id
        GROUP BY u.username
        ORDER BY uploads DESC
        LIMIT 5
    """)
    top_users = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("admin.html",
        total_users=total_users,
        total_datasets=total_datasets,
        total_completed=total_completed,
        new_users_month=new_users_month,
        users=users,
        recent_uploads=recent_uploads,
        uploads_chart=uploads_chart,
        top_users=top_users,
    )


@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    if session.get('user') != 'admin':
        return jsonify({"error": "Access denied"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    # Delete datasets first (foreign key)
    cur.execute("DELETE FROM datasets WHERE user_id = %s", (user_id,))
    cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))

    conn.commit()
    cur.close()
    conn.close()

    session['toast'] = "User deleted successfully."
    return redirect(url_for('admin_dashboard'))


# ---------------- FORGOT PASSWORD ----------------

@app.route('/forgot-password', methods=['GET'])
def forgot_password_page():
    return render_template('forgot_password.html')


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    username = request.form.get('username')

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT email FROM users WHERE username = %s", (username,))
    user = cur.fetchone()

    if not user:
        flash("Username not found.", "error")
        cur.close()
        conn.close()
        return redirect(url_for('forgot_password_page'))

    email = user[0]
    otp = send_otp(email)

    session['reset_otp']  = otp
    session['reset_user'] = username

    flash("OTP sent to your registered email.", "success")
    cur.close()
    conn.close()
    return redirect(url_for('reset_verify_page'))


@app.route('/reset-verify', methods=['GET'])
def reset_verify_page():
    return render_template('verify.html', mode='reset')


@app.route('/reset-verify', methods=['POST'])
def reset_verify():
    entered_otp = request.form.get('otp')
    session_otp = session.get('reset_otp')

    if entered_otp != session_otp:
        flash("Invalid OTP.", "error")
        return render_template('verify.html', mode='reset')

    session.pop('reset_otp', None)
    session['reset_verified'] = True
    return redirect(url_for('new_password_page'))


@app.route('/new-password', methods=['GET'])
def new_password_page():
    if not session.get('reset_verified'):
        return redirect(url_for('home'))
    return render_template('new_password.html')


@app.route('/new-password', methods=['POST'])
def new_password():
    if not session.get('reset_verified'):
        return redirect(url_for('home'))

    new_pass     = request.form.get('password')
    confirm_pass = request.form.get('confirm_password')
    username     = session.get('reset_user')

    if new_pass != confirm_pass:
        flash("Passwords do not match.", "error")
        return render_template('new_password.html')

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("UPDATE users SET password = %s WHERE username = %s", (new_pass, username))
    conn.commit()
    cur.close()
    conn.close()

    session.pop('reset_verified', None)
    session.pop('reset_user', None)

    flash("Password reset successfully! Please login.", "success")
    return redirect(url_for('home'))



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)