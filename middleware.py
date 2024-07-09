import functools
from flask import session, redirect, request, url_for

def auth(view_func):
    @functools.wraps(view_func)
    def decorated(*args, **kwargs):
        if 'email' not in session:
            session['next'] = request.url
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    return decorated

def guest(view_func):
    @functools.wraps(view_func)
    def decorated(*args, **kwargs):
        if 'email' in session:
            return redirect('/dashboard')
        return view_func(*args, **kwargs)
    return decorated


