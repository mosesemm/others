

def check_course_acceptence(subjects, course):

    if not course:
        return False

    if course == "Drawing":
        req_subject = [ subject for subject in subjects if 'science' in subject['subject'].lower()]
        #we only accepting c+ students
        return req_subject[0]['symbol'].lower()  <= 'c' if len(req_subject) > 0 else False

    return True
