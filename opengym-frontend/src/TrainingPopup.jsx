import React from 'react'
const Popup = ({ handleClose }) => {
    return (
    <div className="popup-box">
        <div className="box">
        <span className="close-icon" onClick={handleClose}>x</span>
        {/* Your popup content goes here */}
        <p>This is the popup content!</p>
        </div>
    </div>
    );
};

export default Popup;
