import * as userApis from '../apis/user-api';

export const REQ_REGISTER = 'REQ_REGISTER'
export const REGISTER_RESULTS = 'REGISTER_RESULTS'
export const AUTH_TOKEN = "AUTH_TOKEN"
export const USER_REG_ATTR_CHANGE = 'USER_REG_ATTR_CHANGE'

const reqRegisterUser = () => {
    return {
        type: REQ_REGISTER,
    }
}

const loadRegisterResults = (error) => {

    return {
        type: REGISTER_RESULTS,
        registerError: error
    }
}

const storeToLocalStorage = (key, value) => {

    if(localStorage) {
        localStorage.setItem(key, value);
    }
}

export const getFromLocalStorage = (key) => {

    if(localStorage) {
        return localStorage.getItem(key);
    }
    return "";
}

export const onUserAttrChange = (name, value) => {

    return {
        type: USER_REG_ATTR_CHANGE,
        name,
        value
    }
}

export const registerUser = (password, successCallBack) => {

    return (dispatch, getState) => {

        let user = {...getState().user.userReg, password}
        
        dispatch(reqRegisterUser());
        userApis.registerUser(user).then(
            results => {
                if(results.status == "success") {
                    dispatch(loadRegisterResults(undefined));
                    storeToLocalStorage(AUTH_TOKEN, results.auth_token);
                    successCallBack && successCallBack();
                }else {
                    dispatch(loadRegisterResults(results.message));
                }
            },
            error => {
                dispatch(loadRegisterResults(error.message || error))
            }
        )
        
    }
}