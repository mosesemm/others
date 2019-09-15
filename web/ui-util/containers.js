import React from 'react';

export const Card = ({title, children}) => {

    return <div className="card">
        <div className="card-body">
            <h5 className="mb-4 text-primary">{title}</h5>

            {children}
        </div>
    </div>
}

export const Modal = ({id, children}) => {

    return <div className="modal" role="dialog" id={id}>
        <div className="modal-dialog modal-lg">
            <div className="modal-content">
                {children}
            </div>
        </div>
    </div>
}