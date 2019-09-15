import React, {Component} from 'react';

import ActivePermissions from './active-permissions/active-permissions';
import GrantPermissions from './grant-permissions/grant-permissions';

class App extends Component {

    render() {
        return <div className="container">
            <h2 className="mt-4">Permissions Demo</h2>
            <div className="row">
                <div className="col-sm-12">
                    <GrantPermissions/>
                </div>
            </div>
            <div className="row mt-2">
                <div className="col-sm-12">
                    <ActivePermissions />
                </div>
            </div>
        </div>
    }

}

export default App
