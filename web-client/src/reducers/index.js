
import applicationsReducer from './applications';
import {combineReducers} from 'redux';

export default combineReducers({applications: applicationsReducer})

