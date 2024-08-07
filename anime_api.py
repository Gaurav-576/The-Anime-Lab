import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mysqldb import MySQL
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
import bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)

# Configuration for MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'anime_db'
app.config['SECRET_KEY'] = 'secretkey'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    user = cursor.fetchone()
    cursor.close()
    if user:
        return User(id=user['id'], username=user['username'], email=user['email'], password=user['password'])
    return None


def load_data_and_model():
    global df_anime, df_ratings, user_weights, anime_weights, user_encoder, anime_encoder

    df_anime = pd.read_csv('C:/Users/HP/Major Projects/data/filtered_dataset.csv')
    df_ratings = pd.read_csv('C:/Users/HP/Major Projects/data/ratings.csv')
    with open('C:/Users/HP/Major Projects/data/recommend.pkl', 'rb') as file:
        model = pickle.load(file)
    
    user_encoder = LabelEncoder()
    df_ratings["user_encoded"] = user_encoder.fit_transform(df_ratings["user_id"])
    anime_encoder = LabelEncoder()
    df_ratings["anime_encoded"] = anime_encoder.fit_transform(df_ratings["anime_id"])
    def extract_weights(name, model):
        weight_layer = model.get_layer(name)
        weights = weight_layer.get_weights()[0]
        weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
        return weights

    user_weights = extract_weights('embedding', model)
    anime_weights = extract_weights('embedding_1', model)
    threshold = 1000
    df_anime = df_anime.query('Members >= @threshold')

load_data_and_model()

#homepage
@app.route('/')
def home():
    is_logged_in = current_user.is_authenticated
    featured = [38000, 1735, 40748, 21, 1535]
    action = [269]
    adventure = [4, 52, 113, 13, 146, 82]
    isekai = [76, 84, 95, 114, 45]
    
    featured_animes = df_anime[df_anime['anime_id'].isin(featured)].to_dict(orient='records')
    action_animes = df_anime[df_anime['anime_id'].isin(action)].to_dict(orient='records')
    adventure_animes = df_anime[df_anime['anime_id'].isin(adventure)].to_dict(orient='records')
    isekai_animes = df_anime[df_anime['anime_id'].isin(isekai)].to_dict(orient='records')
    return render_template(
        'index.html',
        is_logged_in=is_logged_in,
        featured=featured_animes,
        action=action_animes,
        adventure=adventure_animes,
        isekai=isekai_animes
    )


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        name = form.name.data
        username = form.username.data
        email = form.email.data
        password = form.password.data

        encrypted_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        try:
            with mysql.connection.cursor() as cursor:
                cursor.execute('INSERT INTO users(name, username, email, password) VALUES(%s, %s, %s, %s)', (name, username, email, encrypted_password))
                mysql.connection.commit()
                flash("Registration successful! Please login to access your details.", "success")
                return redirect(url_for('login'))
        except mysql.connection.Error as e:
            if e.args[0] == 1062:
                flash("Username or email already exists. Please choose a different one.", "error")
            else:
                flash(f"An error occurred: {e}", 'error')
            return redirect(url_for('register'))
    return render_template('register.html', form=form)

# Login form
class LoginForm(FlaskForm):
    login_credentials = StringField('Username or Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        login_credentials = form.login_credentials.data
        password = form.password.data
        
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username=%s OR email=%s', (login_credentials, login_credentials))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            flash("Login successful!", 'success')
            user_obj = User(id=user['id'], username=user['username'], email=user['email'], password=user['password'])
            login_user(user_obj)
            return redirect(url_for('mylist'))
        else:
            flash("Login or password is incorrect", 'error')
            return redirect(url_for('login'))
    return render_template('login.html', form=form)

# recommendation logic
def item_item_collaborative_filtering(name, n, return_dist=False, neg=False):
    try:
        anime_row=df_anime[df_anime['Name'] == name].iloc[0]
        idx=anime_row['anime_id']
        encoded_idx=anime_encoder.transform([idx])[0]
        weights=anime_weights
        dist=np.dot(weights,weights[encoded_idx]) # cosine similarity
        sorted_dist=np.argsort(dist)
        n=n+1
        if neg:
            similar=sorted_dist[:n]
        else:
            similar=sorted_dist[-n:]
        if return_dist:
            return dist,similar
            
        sim_arr=[]
        for sim in similar:
            decoded_id=anime_encoder.inverse_transform([sim])[0]
            anime_frame=df_anime[df_anime['anime_id']==decoded_id]
            anime_id=anime_frame['anime_id'].values[0]
            anime_name=anime_frame['Name'].values[0]
            english_name=anime_frame['English name'].values[0]
            img_id=anime_frame['Image URL'].values[0]
            name=english_name if english_name!="UNKNOWN" else anime_name
            similarity=dist[sim]
            similarity="{:.2f}%".format(similarity*100)
            sim_arr.append({"anime_id":anime_id,"Name":name,"Similarity":similarity,"Image URL":img_id})
        sim_df=pd.DataFrame(sim_arr).sort_values(by="Similarity",ascending=False)
        return sim_df[sim_df.Name!=name]
    except:
        print('{} not found in Anime list'.format(name))

#load anime
@app.route('/anime/<int:anime_id>')
def anime(anime_id):
    anime=df_anime[df_anime['anime_id'] == anime_id].iloc[0]
    recommendations=item_item_collaborative_filtering(anime['Name'],20)
    recommendations=recommendations.to_dict(orient='records')
    return render_template('animePage.html', anime=anime, recommendations=recommendations)


@app.route('/animeList')
def anime_list():
    return render_template('animeList.html')

@app.route('/mangaList/')
def manga_list():
    return render_template('mangaList.html')

# Search route
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if query:
        filtered_df = df_anime[df_anime['Name'].str.contains(query, case=False, na=False)]
        search_results = filtered_df.to_dict(orient='records')
    else:
        search_results = []
    return render_template('search_results.html', results=search_results)

@app.route('/add_to_wishlist', methods=['POST'])
def add_to_wishlist():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'Please login to add anime to your wishlist'}), 401

    user_id = current_user.id
    anime_id = request.form['anime_id']
    status = request.form['status']
    
    with mysql.connection.cursor() as cursor:
        cursor.execute('INSERT INTO animelist (user_id, anime_id, status) VALUES (%s, %s, %s)', (user_id, anime_id, status))
        mysql.connection.commit()
        cursor.close()
    return jsonify({'status': 'success', 'message': 'Anime added to wishlist'})

# MyList route
@app.route('/mylist')
@login_required
def mylist():
    form = LogoutForm()
    return render_template('mylist.html', user=current_user, form=form)

# Logout form for CSRF protection
class LogoutForm(FlaskForm):
    submit = SubmitField('Logout')

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()  # Log out the current user
    session.clear()  # Clear the session
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))  # Redirect to the homepage

if __name__ == '__main__':
    app.run(debug=True)
