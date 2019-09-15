import React, {Component} from 'react';

//hopefully get a better way to include all thise
import * as $ from 'jquery';
import * as datatable from "datatables.net-bs4";
import * as responsive_datatable from "datatables.net-responsive";

import {Card, Modal} from '../ui-util/containers';

class ActivePermissions extends Component {

    constructor() {
        super();
    }

    componentDidMount() {

        const test = [
            {dateGranted: "15/09/2019", grantedBy: "Mosd Test", grantedTo: "Sipho Test", effectiveRole: ""},
            {dateGranted: "15/09/2019", grantedBy: "Lebo Test", grantedTo: "Lerato Test", effectiveRole: ""},
            {dateGranted: "15/09/2019", grantedBy: "Megail Test", grantedTo: "Thabo Test", effectiveRole: ""}
        ];

        $("#active_permissions").DataTable({
            data: test,
            iDisplayLength: 10,
            bLengthChange: false,
            responsive: true,
            columns: [
                {data: "dateGranted", title: "Date granted", width:"20%"},
                {data: "grantedBy", title: "Granted by"},
                {data: "grantedTo", title: "Recipient"},
                {data: "grantedTo", title: "View", width:"10%", render: (data, type, row) => {
                        return "<a href='#' data-toggle='modal' class='text-primary' " +
                            "data-target='#view-contract'> <i class='fa fa-sticky-note'></i>";
                }},
                {data: "grantedTo", title: "Revoke", width:"10%", render: (data, type, row) => {
                    return "<a href='#' data-toggle='modal' class='text-danger' " +
                        "data-target='#view-contract'> <i class='fa fa-times-circle'></i>";
                }}
            ]
        });

    }

    render() {

        return <Card title="Active Permissions">
            <table id="active_permissions" className="table table-striped table-bordered"></table>

            <Modal id="view-contract">
                <div className="modal-header">
                    <h5 className="modal-title">Terms and conditions</h5>
                </div>
                <div className="modal-body">
                    fasdf ags g sdg fsg  sd sfgsggfgfg gs g fg  g gsgfgs fg s  dfg sd fg fg sg f g sgsgs
                    gs gs fdgsf g  g sg fsdg  g dfg  sg s g  fdsg s g s s g  s fdg d fg s
                </div>
                <div className="modal-footer">
                    <button className="btn btn-secondary" data-dismiss="modal">Close</button>
                    <button className="btn btn-primary">Confirm</button>
                </div>
            </Modal>
        </Card>
    }
}

export default ActivePermissions