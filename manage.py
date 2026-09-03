#!/usr/bin/env python
"""Django's command-line utility for administrative tasks.

Usage:
    python manage.py runserver 8005     # Start development server on port 8005
    python manage.py makemigrations     # Create new database migration files
    python manage.py migrate            # Apply pending migrations to MySQL
    python manage.py createsuperuser    # Create admin panel superuser account
    python manage.py test               # Run all unit and integration tests
"""
import os
import sys

def main():
    """Run administrative tasks.
    
    Sets the DJANGO_SETTINGS_MODULE environment variable and delegates
    to Django's management command infrastructure.
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'loan_prediction.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()