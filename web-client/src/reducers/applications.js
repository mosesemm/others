
import {STEP_TO} from '../actions/applications';
const intialState = {
    step: 1
}

const applicationsReducer = (state = intialState, action) => {

    switch(action.type) {
        case STEP_TO:
            return {...state, step:action.step}
        default:
            return state;
    }

}

export default applicationsReducer