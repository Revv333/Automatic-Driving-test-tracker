import React from 'react';
import { Link } from 'react-router-dom';
import Navbar from './Navbar';

function Home() {
    return (
        <div style={{ paddingTop: '60px' }}>
            <Navbar/>
            <h1>Home</h1>
            <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Suscipit facilis adipisci veritatis dicta. Consequuntur vero repellendus modi, soluta dolores asperiores. Sit pariatur voluptatum natus tempore assumenda atque ea aperiam ducimus!</p>
            <h1>Home</h1>
            <Link to="/fv" className="btn btn-dark">Apply For License!!</Link>
        </div>
    );
}

export default Home;
