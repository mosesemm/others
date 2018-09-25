
import * as api from '../apis/applications-api';
export const STEP_TO="STEP_TO";
export const SET_CURRENT_FILE="SET_CURRENT_FILE";
export const REQ_PARSE_CERT = "REQ_PARSE_CERT";
export const SET_PARSE_RESULTS = "SET_PARSE_RESULTS";
export const SET_SUBJECT_ITEM = "SET_SUBJECT_ITEM";

export const goToStep = (step) => {

    return {
        step,
        type: STEP_TO
    }
}

export const setCurrentFile = (file) => {

    return {
        type: SET_CURRENT_FILE,
        file: file
    }
}

export const requestParsCert = () => {

    return {
        type: REQ_PARSE_CERT,
    }
}

export const setParseResults = (subjects, examNumber, parseErrors) => {

    return {
        type: SET_PARSE_RESULTS,
        subjects,
        examNumber,
        parseErrors
    }
}

export const subjectItemChange = (key, value) => {

    return {
        type: SET_SUBJECT_ITEM,
        key,
        value
    }
}

export const fileOnChange = (file) => {

    return (dispatch, getState) => {
        dispatch(setCurrentFile(file));
        dispatch(requestParsCert());
        api.parseCertificate(file).then(
            (results) => {
                if (results && results.status == 'success' && results.data){
                    dispatch(setParseResults(results.data.subjects, results.data.examNumber));
                }
                else{
                    dispatch(setParseResults(undefined, undefined, results.message));
                }
            },
            error => {
                console.log(error);
                dispatch(setParseResults(undefined, undefined, error.message));
            }
        )
    }
}