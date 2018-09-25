import React, {Component} from 'react';
import {Well, FormGroup, FormControl, ControlLabel, Button} from 'react-bootstrap';

const Login = ({history}) => {


    return (
        <Well>
            <form>
                <FormGroup>
                    <ControlLabel>Email</ControlLabel>
                    <FormControl type="text"/>
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Password</ControlLabel>
                    <FormControl type="text"/>
                </FormGroup>
                <Button type="button" onClick={() => history.push('/register')} className="push--right-5">Register</Button>
                <Button type="button">Login</Button>
            </form>
        </Well>
    )
}

export default Login