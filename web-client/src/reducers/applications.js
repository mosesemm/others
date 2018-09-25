
import {STEP_TO, SET_CURRENT_FILE, REQ_PARSE_CERT, SET_PARSE_RESULTS, SET_SUBJECT_ITEM} from '../actions/applications';
const intialState = {
    step: 1,
    req_loading: false,
    subjects: []
}

const applicationsReducer = (state = intialState, action) => {

    switch(action.type) {
        case STEP_TO:
            return {...state, step:action.step};
        case SET_CURRENT_FILE:
            return {...state, file: action.file};
        case REQ_PARSE_CERT:
            return {...state, req_loading: true};
        case SET_PARSE_RESULTS:
            return {...state, req_loading: false, subjects: action.subjects || [], examNumber: action.examNumber,
                    parseErrors: action.parseErrors};
        case SET_SUBJECT_ITEM:
                return {...state, subjects: changeSubject(state.subjects, action.key, action.value)}
        default:
            return state;
    }

}

const changeSubject = (subjects, key, value) => {
    
    return subjects.map(item => {
        if (key in Object.keys(item)){
            item[key] = value;
        }
        if (('symbol_'+key) in Object.keys(item)){
            item['symbol_'+key] = value;
        }

        return item;
    })
}

export default applicationsReducer