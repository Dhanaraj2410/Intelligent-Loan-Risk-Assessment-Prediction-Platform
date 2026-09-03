# =============================================================================
# __init__.py - Django Project Package Initialization
# =============================================================================
# Configures PyMySQL as the MySQL database adapter for Django.
# Django expects MySQLdb (mysqlclient) by default, but PyMySQL is a
# pure-Python alternative that is easier to install on Windows systems.
# The install_as_MySQLdb() call patches PyMySQL to be compatible with
# Django's MySQLdb interface.
# =============================================================================

import pymysql
pymysql.install_as_MySQLdb()