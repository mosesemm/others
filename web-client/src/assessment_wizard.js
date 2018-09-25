
import React, {Component} from 'react';
import {Well, ControlLabel, Button, FormControl, FormGroup, Panel, Row} from 'react-bootstrap';
import {connect} from 'react-redux';
import {goToStep, fileOnChange, subjectItemChange} from './actions/applications';
import loader from './img/loading.gif';

export const AssessmentStep1 = ({goToStep, fileOnChange, req_loading, parseErrors, subjects, subjectItemChange}) => {


    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Required info </h4>
                <hr/>
                {
                    parseErrors? <div className="alert alert-danger">{parseErrors}</div>:''
                }
                <FormGroup>
                    <ControlLabel> Upload matric certificate </ControlLabel>
                    <input type="file" onChange={(event) => {console.log(event.target.files[0]); fileOnChange(event.target.files[0])}}/>
                    {   
                        req_loading? <div><img src={loader} alt="Loading..."/></div>:''
                    }
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Exam number</ControlLabel>
                    <FormControl type="text" />
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

                <hr/>

                <Button type="button" className="push--right-5">Cancel</Button>
                <Button type="button" onClick = {() => goToStep(2)} >Next</Button>
            </div>
        </div>
    )
}

export const AssessmentStep2 = ({goToStep}) => {

    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Confirmation</h4>
                <hr/>
                <Panel>
                    <table className="table table-hover table-bordered">
                        <tbody>
                            <tr className="warning">
                                <th>
                                    Certificate verification
                                </th>
                                <td>
                                    Ok
                                </td>
                            </tr>
                            <tr className="success">
                                <th>
                                    Course verification
                                </th>
                                <td>
                                    Ok
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

const FinalStep = ({goToStep, history}) => {

    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Thank you</h4>
                <hr/>
                <div className="alert alert-success">
                    <p>You have been accepted for the course applied</p>
                </div>
                <hr/>
                <Button type="button" onClick= {() => history.push('/')}> Done </Button>
            </div>
        </div>
    )
}

const ApplicationAssessment = ({step, goToStep, history, fileOnChange, req_loading, parseErrors, subjects, subjectItemChange}) => {

    return (
        <div>
            <h5>Step {step} / 3 </h5>
            {
                step == 1? <AssessmentStep1     fileOnChange = {fileOnChange}
                                                goToStep = {goToStep}
                                                req_loading = {req_loading}
                                                parseErrors = {parseErrors}
                                                subjects = {subjects} 
                                                subjectItemChange = {subjectItemChange} />
                : step == 2? <AssessmentStep2 goToStep = {goToStep}/>
                : step == 3? <FinalStep history = {history}
                                        goToStep = {goToStep}/> : ''

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

    }},
    (dispatch) => {

        return {
             goToStep: (step) => dispatch(goToStep(step)),
             fileOnChange: (file) => dispatch(fileOnChange(file)),
             subjectItemChange: (key, value) => dispatch(subjectItemChange(key, value))
        }
    }
)(ApplicationAssessment)

export default ApplicationAssessmentContainer
