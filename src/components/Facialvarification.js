import axios from 'axios';
import React from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from './Navbar';

function Facialvarification() {
    const navigate=useNavigate();
    const startverify = async (e) => {
        e.preventDefault();
        try{
            const response= await axios.post('http://127.0.0.1:5000/fv');
            console.log(response.data.Verification);
            if(response.data.Verification==="True"){
                navigate('/upload');
            }
        }
        catch(error){
            console.log(error);
        }
    }
    return (
        <div style={{ paddingTop: '60px' }}>
            <Navbar/>
            <button className='btn btn-dark' onClick={startverify}>Verification</button>
            {/* <Link to="/upload" className="btn btn-dark">UploadVideo</Link> */}
        </div>
    )
}

export default Facialvarification
