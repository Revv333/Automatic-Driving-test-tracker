import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import Navbar from './Navbar';

const VideoUpload = () => {
    const [videoFile, setVideoFile] = useState(null);
    const [coordinateFile, setCoordinateFile] = useState(null);
    const [message, setMessage] = useState('');

    // useEffect(() => {
    //     // Check if the user is logged in or authenticated (you may need to adjust this)
    //     const isAuthenticated = localStorage.getItem('authenticated');
    //     if (!isAuthenticated) {
    //         setMessage('Please log in before uploading');
    //     }
    // }, []);

    const handleVideoFileChange = (event) => {
        setVideoFile(event.target.files[0]);
    };

    const handleCoordinateFileChange = (event) => {
        setCoordinateFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        
        if (videoFile && coordinateFile) {
            formData.append('video', videoFile);
            formData.append('coordinate', coordinateFile);
            
            try {
                const token = localStorage.getItem('authenticated'); // Get the authentication token
                const config = {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                        'Authorization': `Bearer ${token}`, // Include the authentication token in headers
                    },
                };
                
                const response = await axios.post('http://127.0.0.1:5000/upload-data', formData, config);
                console.log(response.data); // Handle the response from the backend
            } catch (error) {
                console.error('Error uploading the files:', error);
                setMessage('Login Before Uploading');
                console.log(message);
            }
        } else {
            setMessage('Please select both video and coordinate files');
        }
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <Navbar/>
            <div className='card' style={{ width: '25rem' }}>
                <div className='card-body'>
                    {/* {message && <p>{message}</p>} */}
                    <form onSubmit={handleSubmit}>
                        <div className="mb-3">
                            <label htmlFor="videoUpload" className="form-label">Upload Your Video</label>
                            <input
                                type="file"
                                className="form-control"
                                id="videoUpload"
                                accept=".mp4"
                                onChange={handleVideoFileChange}
                            />
                        </div>

                        <div className="mb-3">
                            <label htmlFor="coordinateUpload" className="form-label">Upload Your Coordinate file</label>
                            <input
                                type="file"
                                className="form-control"
                                id="coordinateUpload"
                                accept=".csv"
                                onChange={handleCoordinateFileChange}
                            />
                        </div>

                        <button type="submit" className="btn btn-dark" style={{ margin: '10px' }}>Upload</button>
                        <Link to="/result" className="btn btn-dark">Get Result</Link>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default VideoUpload;
