import React, { Component } from 'react';
import Register from './register';
import Home from './home';
import { BrowserRouter, Route, Link, Switch } from 'react-router-dom'
import ApplicationAssessmentContainer from './assessment_wizard'
import rootReducer from './reducers'
import {createStore, applyMiddleware, compose} from 'redux';
import {Provider} from 'react-redux';
import Login from './login';

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
const store = createStore(rootReducer, composeEnhancers());



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
          {/**<Link to="/" className="navbar-brand">CSA</Link>**/}

          <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
            <ul className="nav navbar-nav">
              <li className="active"><Link to="/">Home</Link></li>
              <li><Link to="/applications">Applications</Link></li>
            </ul>
            
            <ul className="nav navbar-nav navbar-right">
              <li><Link to="#">Logout</Link></li>
            </ul>
            
          </div>
        </div>
        </div>
    </nav>  
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
              <h3>Academic application assessment</h3>  
            </div>
            <div className="container">
              <Switch>
                <Route exact path="/" component={Home}/>
                <Route path="/login" component={Login} />
                <Route path="/applications" component={ApplicationAssessmentContainer}/>
                <Route path="/register" component={Register} />
              </Switch>
            </div>
          </div>
        </BrowserRouter>
      </Provider>
    );
  }
}

export default App;
