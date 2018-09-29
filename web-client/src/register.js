import React, {Component} from 'react';
import {Well,ControlLabel ,FormControl, FormGroup, Button} from 'react-bootstrap';
import {connect} from 'react-redux';
import {onUserAttrChange, registerUser} from './actions/user-actions';

const Register = ({user, onUserAttrChange, registerError, registerUser, history}) => {

    let errors = registerError;

    let passInput = React.createRef();
    let confirmPassInput = React.createRef();

    return (
        <Well>

            {
                errors? <div className="alert alert-danger">{errors}</div>:''
            }
            <form>
                <FormGroup>
                    <ControlLabel>Name</ControlLabel>
                    <FormControl type="text" name="name" 
                                            value={user.name}
                                            onChange = {event => onUserAttrChange(event.target.name, event.target.value)}/>
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Surname</ControlLabel>
                    <FormControl type="text" name="surname"
                                            value = {user.surname}
                                            onChange = {event => onUserAttrChange(event.target.name, event.target.value)}/>
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Email</ControlLabel>
                    <FormControl type="text" name="email"
                                            value={user.email}
                                            onChange={event => onUserAttrChange(event.target.name, event.target.value)}/>
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Password</ControlLabel>
                    <input type="password" name="password" ref={passInput} className="form-control"/>
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Confirm password</ControlLabel>
                    <input type="password" name="confirmPassword" ref={confirmPassInput} className="form-control"/>
                </FormGroup>
                <Button type="reset" className="push--right-5">Reset</Button>
                <Button type="button" onClick={ () => {
                        if(passInput.current.value != confirmPassInput.current.value) {
                            //this is not working will fix later
                            errors = "Password and confirm password must be the same";
                            return; 
                        }
                        registerUser(passInput.current.value, () => {
                        console.log('Successfully logged in, maybe display some global message');
                        history.push('/');})}
                    }>Submit</Button>
            </form>
        </Well>
    )
}

const RegisterContainer = connect((state) => {
    return {
        user: state.user.userReg,
        registerError: state.user.registerError,
    }},
    (dispatch) => {
        return {
            onUserAttrChange: (name, value) => dispatch(onUserAttrChange(name, value)),
            registerUser: (password, callBack) => dispatch(registerUser(password, callBack))
        }
    }

) (Register)

export default RegisterContainer