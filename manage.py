import os
import unittest
import coverage

from flask_script import Manager

from server import app

cov = coverage.coverage(
    branch = True,
    include = 'server/*',
    omit = [
        'tests/*',
        'server/config.py',
        'server/__init__.py'
    ]
)

cov.start()

manager = Manager(app)


@manager.command
def test():
    result = run_tests()
    if result.wasSuccessful():
        return 0
    return  1

@manager.command
def coverage():
    result = run_tests()
    if result.wasSuccessful():
        cov.stop()
        cov.save()
        print('Coverage Summary: ')
        cov.report()
        basedir = os.path.abspath(os.path.dirname(__file__))
        covdir = os.path.join(basedir, 'tmp/coverage')
        cov.html_report(directory=covdir)
        print('html version: file://%s/index.html' % covdir)
        cov.erase()
        return 0

    return 1

def run_tests():
    tests = unittest.TestLoader().discover('tests', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return result

if __name__ == "__main__":
    manager.run()
