
import React, {Component} from 'react';
import {Well, ControlLabel, Button, FormControl, FormGroup, Panel} from 'react-bootstrap';
import {connect} from 'react-redux';
import {goToStep} from './actions/applications';

export const AssessmentStep1 = ({goToStep}) => {


    return (
        <div className="panel panel-default">
            <div className="panel-body">
                <h4>Required info </h4>
                <hr/>
                <FormGroup>
                    <ControlLabel> Upload matric certificate </ControlLabel>
                    <input type="file" />
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Exam number</ControlLabel>
                    <FormControl type="text" />
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Subject</ControlLabel>
                    <FormControl type="text" />
                </FormGroup>

                <hr/>

                <Button type="button">Cancel</Button>
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
                <Button type="button" onClick = {() => goToStep(1)}>Back</Button>
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

const ApplicationAssessment = ({step, goToStep, history}) => {

    return (
        <div>
            <h5>Step {step} / 3 </h5>
            {
                step == 1? <AssessmentStep1  goToStep = {goToStep}/>
                : step == 2? <AssessmentStep2 goToStep = {goToStep}/>
                : step == 3? <FinalStep history = {history}
                                        goToStep = {goToStep}/> : ''

            }
        </div>
    )
}

const ApplicationAssessmentContainer = connect((state) => {
    return {
        step: state.applications.step
    }},
    (dispatch) => {

        return {
             goToStep: (step) => dispatch(goToStep(step)),
        }
    }
)(ApplicationAssessment)

export default ApplicationAssessmentContainer
