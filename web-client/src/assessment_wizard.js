
import React, {Component} from 'react';
import {Well, ControlLabel, Button, FormControl, FormGroup, Panel, Row} from 'react-bootstrap';
import {connect} from 'react-redux';
import {goToStep, fileOnChange, subjectItemChange, onGenericAttrChange, submitAssessCourseOrCert} from './actions/applications';
import loader from './img/loading.gif';

export const AssessmentStep1 = ({goToStep, fileOnChange, req_loading, parseErrors, subjects, subjectItemChange, examNumber, 
                    verificationErrors, req_ass_loading,
                    course, onGenericAttrChange, submitAssessCourseOrCert}) => {


    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Required info </h4>
                <hr/>
                {
                    parseErrors || verificationErrors? <div className="alert alert-danger">{parseErrors || verificationErrors}</div>:''
                }
                <FormGroup>
                    <ControlLabel> Upload matric certificate </ControlLabel>
                    <input type="file" onChange={(event) => {console.log(event.target.files[0]); fileOnChange(event.target.files[0])}}/>
                    <p>
                    {   
                        req_loading? <div><i className="fas fa-spinner push--right-5 "></i>Loading...</div>:''
                    }
                    </p>
                </FormGroup>

                <Well>
                    <FormGroup>
                        <ControlLabel>Exam number</ControlLabel>
                        <FormControl type="text" name="examNumber" value={examNumber} 
                                                                onChange={(event) => onGenericAttrChange(event.target.name, event.target.value)}/>
                    </FormGroup>

                    <table className="table">
                        <thead>
                            <th>Subject</th>
                            <th>Symbol</th>
                        </thead>
                        <tbody>
                        {
                            subjects.map(item => <tr key={item.subject}>
                                <td><FormControl value={item.subject || ""}
                                                onChange={(event) => subjectItemChange(event.target.name, event.target.value)}
                                                type="text" name={item.subject}/></td>
                                <td><FormControl value={item.symbol || ""}
                                                onChange={(event) => subjectItemChange(event.target.name, event.target.value)}
                                                type="text" name={'symbol_'+item.subject}/></td>
                            </tr>)
                        }
                        </tbody>
                    </table>
                </Well>

                <Well>
                    <FormGroup>
                        <ControlLabel >Course </ControlLabel>
                        <select name="course" value={course} 
                                                onChange={(event) => onGenericAttrChange(event.target.name, event.target.value)}
                                                className="form-control" >
                            <option value="">Please select</option>
                            <option value="Accounting">Accounting</option>
                            <option value="Drawing">Drawing</option>
                            <option value="Economics">Economics</option>
                        </select>
                    </FormGroup>

                </Well>

                <hr/>

                <Button type="button" className="push--right-5">Cancel</Button>
                <Button type="button" onClick = {() => submitAssessCourseOrCert()} disabled={req_ass_loading}>{req_ass_loading?'Loading...':'Next'}</Button>
            </div>
        </div>
    )
}

export const AssessmentStep2 = ({goToStep, certVerification, courseVerification}) => {

    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Confirmation</h4>
                <hr/>
                <Panel>
                    <table className="table table-hover table-bordered">
                        <tbody>
                            <tr className={certVerification?'success': 'warning'}>
                                <th>
                                    Certificate verification
                                </th>
                                <td>
                                    {certVerification?'Ok': 'Not Verified'}
                                </td>
                            </tr>
                            <tr className={courseVerification?'success':'warning'}>
                                <th>
                                    Course verification
                                </th>
                                <td>
                                    {courseVerification?'Ok':"Not Verified"}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </Panel>
                <hr/>
                <Button type="button" onClick = {() => goToStep(1)} className="push--right-5">Back</Button>
                <Button type="button" onClick={() => goToStep(3)}>Submit</Button>
            </div>
        </div>
    )
}

const FinalStep = ({goToStep, history, certVerification, courseVerification}) => {

    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Thank you</h4>
                <hr/>
                {
                    certVerification && courseVerification? 
                <div className="alert alert-success">
                    <p>You have been accepted for the course applied</p>
                </div>:
                <div className="alert alert-warning">
                <p>Your applicaiton has been submitted for further checks</p>
            </div>
                }
                <hr/>
                <Button type="button" onClick= {() => history.push('/')}> Done </Button>
            </div>
        </div>
    )
}

const ApplicationAssessment = ({step, goToStep, history, fileOnChange, req_loading, parseErrors, subjects, subjectItemChange, examNumber,req_ass_loading, 
                            course,onGenericAttrChange, submitAssessCourseOrCert, verificationErrors, certVerification, courseVerification}) => {

    return (
        <div>
            <h5>Step {step} / 3 </h5>
            {
                step == 1? <AssessmentStep1     fileOnChange = {fileOnChange}
                                                goToStep = {goToStep}
                                                req_loading = {req_loading}
                                                parseErrors = {parseErrors}
                                                subjects = {subjects} 
                                                examNumber = {examNumber}
                                                course = {course}
                                                verificationErrors = {verificationErrors}
                                                subjectItemChange = {subjectItemChange}
                                                onGenericAttrChange = {onGenericAttrChange}
                                                req_ass_loading = {req_ass_loading}
                                                submitAssessCourseOrCert = {submitAssessCourseOrCert} />

                : step == 2? <AssessmentStep2 goToStep = {goToStep}
                                            certVerification = {certVerification}
                                            courseVerification = {courseVerification}/>
                : step == 3? <FinalStep history = {history}
                                        goToStep = {goToStep}
                                        certVerification = {certVerification}
                                        courseVerification = {courseVerification}/> : ''

            }
        </div>
    )
}

const ApplicationAssessmentContainer = connect((state) => {
    return {
        step: state.applications.step,
        req_loading: state.applications.req_loading,
        parseErrors: state.applications.parseErrors,
        subjects: state.applications.subjects,
        examNumber: state.applications.examNumber,
        course: state.applications.course,
        certVerification: state.applications.certVerification,
        verificationErrors: state.applications.verificationErrors,
        courseVerification: state.applications.courseVerification,
        req_ass_loading: state.applications.req_ass_loading

    }},
    (dispatch) => {

        return {
             goToStep: (step) => dispatch(goToStep(step)),
             fileOnChange: (file) => dispatch(fileOnChange(file)),
             subjectItemChange: (key, value) => dispatch(subjectItemChange(key, value)),
             onGenericAttrChange: (key, value) => dispatch(onGenericAttrChange(key, value)),
             submitAssessCourseOrCert: () => dispatch(submitAssessCourseOrCert())
        }
    }
)(ApplicationAssessment)

export default ApplicationAssessmentContainer
