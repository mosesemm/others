import React, { Component } from 'react';
import RegisterContainer from './register';
import Home from './home';
import { BrowserRouter, Route, Link, Switch, NavLink } from 'react-router-dom'
import ApplicationAssessmentContainer from './assessment_wizard'
import rootReducer from './reducers'
import {createStore, applyMiddleware, compose} from 'redux';
import {Provider} from 'react-redux';
import Login from './login';
import thunk from 'redux-thunk';

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
const store = createStore(rootReducer, composeEnhancers(
  applyMiddleware(
     thunk, /*, loggerMiddleware*/)
));



const Nav = () => {

  return (
    <nav className="navbar navbar-default navbar-fixed-top">
      <div className="container">
        <div className="navbar-header">
          <button type="button" className="navbar-toggle" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
            <span className="sr-only">Toggle navigation</span>
            <span className="icon-bar"></span>
            <span className="icon-bar"></span>
            <span className="icon-bar"></span>
          </button>
          {<Link to="/" className="navbar-brand">CSA</Link>}
        </div>

          <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
            <ul className="nav navbar-nav">
              <li><NavLink to="/">Home</NavLink></li>
              <li><NavLink to="/applications">Applications</NavLink></li>
            </ul>
            
            <ul className="nav navbar-nav navbar-right">
              <li className="dropdown">
                <a className="dropdown-toggle" data-toggle="dropdown">User name <b className="caret"></b></a>
                <ul className="dropdown-menu">
                  <li><NavLink to="/api/logout">Logout</NavLink></li>
                </ul>
             </li>
            </ul>
          </div>
        </div>
    </nav>  
  )
}

const Footer = () => {

  return (
    <div className="container">
        <p> &#169; 2018 Wits Computer Science department</p>
    </div>
  )
}

class App extends Component {
  render() {
    return (
      <Provider store={store}>
        <BrowserRouter>
          <div className="container-fluid">
            <div className="row">
              <Nav />
            </div>
            <div className="jumbotron">
              <h2>College Application Assessment</h2>  
            </div>
            <div className="container">
              <Switch>
                <Route exact path="/" component={Home}/>
                <Route path="/login" component={Login} />
                <Route path="/applications" component={ApplicationAssessmentContainer}/>
                <Route path="/register" component={RegisterContainer} />
              </Switch>
            </div>
            <hr/>
            <Footer />
          </div>
        </BrowserRouter>
      </Provider>
    );
  }
}

export default App;
