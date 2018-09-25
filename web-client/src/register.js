import React, {Component} from 'react';
import {Well,ControlLabel ,FormControl, FormGroup, Button} from 'react-bootstrap';

const Register = () => {

    return (
        <Well>
            <form>
                <FormGroup>
                    <ControlLabel>Name</ControlLabel>
                    <FormControl type="text" />
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Surname</ControlLabel>
                    <FormControl type="text" />
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Email</ControlLabel>
                    <FormControl type="text" />
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Password</ControlLabel>
                    <FormControl type="password" />
                </FormGroup>
                <FormGroup>
                    <ControlLabel>Confirm password</ControlLabel>
                    <FormControl type="password" />
                </FormGroup>
                <Button type="reset">Reset</Button>
                <Button type="button">Submit</Button>
            </form>
        </Well>
    )
}

export default Register