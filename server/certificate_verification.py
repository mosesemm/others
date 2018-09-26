

def verify_certificate(exam_number, subjects):
    #this is just a dummy services, probably it will need to call some external service
    # with some matric certicate verifying authority

    return True if exam_number and len(exam_number) > 5 else False
