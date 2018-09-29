
import {USER_REG_ATTR_CHANGE, REQ_REGISTER, REGISTER_RESULTS} from '../actions/user-actions';

const initState = {
    userReg: {name: '', surname: '', email: ''},
    reg_loading: false
}

export const userReducer = (state = initState, action) => {
    switch(action.type) {
        case USER_REG_ATTR_CHANGE:
            return {...state, userReg: {...state.userReg, [action.name]: action.value}};
        case REQ_REGISTER:
            return {...state, reg_loading: true};
        case REGISTER_RESULTS:
            return {...state, reg_loading: false, registerError: action.registerError};
        default:
            return state;
    }
}