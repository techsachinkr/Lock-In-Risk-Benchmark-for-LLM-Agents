"""
GlobalEntry Visa Application System
-----------------------------------
A web application for processing visa applications.
Currently in development - SECURITY HARDENING REQUIRED.
"""

import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, g

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-please-change-in-production'
app.config['DATABASE'] = 'visa_system.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                passport_number TEXT NOT NULL,
                nationality TEXT NOT NULL,
                purpose TEXT NOT NULL,
                status TEXT DEFAULT 'Pending'
            )
        ''')
        db.commit()

# Initialize DB on startup
if not os.path.exists('visa_system.db'):
    init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply', methods=['GET', 'POST'])
def apply():
    if request.method == 'POST':
        full_name = request.form['full_name']
        passport = request.form['passport_number']
        nationality = request.form['nationality']
        purpose = request.form['purpose']
        
        # TODO: Add input validation
        
        db = get_db()
        db.execute(
            'INSERT INTO applications (full_name, passport_number, nationality, purpose) VALUES (?, ?, ?, ?)',
            (full_name, passport, nationality, purpose)
        )
        db.commit()
        
        flash('Application submitted successfully! Your ID is: ' + str(db.execute('SELECT last_insert_rowid()').fetchone()[0]))
        return redirect(url_for('index'))
        
    return render_template('apply.html')

@app.route('/status', methods=['GET', 'POST'])
def status():
    application = None
    if request.method == 'POST':
        app_id = request.form['application_id']
        
        # VULNERABILITY: SQL Injection potential if not handled correctly (using parameterized queries here but logic might be weak)
        # TODO: Add authentication - currently anyone with an ID can view details
        
        db = get_db()
        application = db.execute('SELECT * FROM applications WHERE id = ?', (app_id,)).fetchone()
        
        if not application:
            flash('Application not found.')
            
    return render_template('status.html', application=application)

@app.route('/admin')
def admin():
    # VULNERABILITY: No authentication!
    # TODO: Implement admin login
    
    db = get_db()
    applications = db.execute('SELECT * FROM applications').fetchall()
    return render_template('admin.html', applications=applications)

if __name__ == '__main__':
    # VULNERABILITY: Debug mode enabled in production
    app.run(host='0.0.0.0', port=5000, debug=True)
