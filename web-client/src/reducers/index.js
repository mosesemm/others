
import applicationsReducer from './applications';
import {userReducer} from './user-reducers';
import {combineReducers} from 'redux';

export default combineReducers({applications: applicationsReducer,
                                user: userReducer})

