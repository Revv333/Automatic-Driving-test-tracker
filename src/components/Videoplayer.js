import React, { useState, useEffect } from 'react';
import Navbar from './Navbar';

function VideoPlayer() {
    // const [videoSrc, setVideoSrc] = useState('');
    const [imageSrc, setImageSrc] = useState('');

    useEffect(() => {
        async function fetchMedia() {
            try {
                // const videoResponse = await fetch('http://127.0.0.1:5000/getresultvideo/video');
                const imageResponse = await fetch('http://127.0.0.1:5000/getresultvideo/image');

                // if (videoResponse.ok) {
                //     console.log(videoResponse)
    
                //     const videoBlob = await videoResponse.blob();
                //     const videoURL = URL.createObjectURL(videoBlob);
                //     setVideoSrc(videoURL);
                // } else {
                //     console.error('Failed to fetch the video');
                // }

                if (imageResponse.ok) {
                    console.log(imageResponse)
                    const imageBlob = await imageResponse.blob();
                    const imageURL = URL.createObjectURL(imageBlob);
                    setImageSrc(imageURL);
                } else {
                    console.error('Failed to fetch the image');
                }
            } catch (error) {
                console.error('Error fetching media:', error);
            }
        }
        fetchMedia();
    }, []);

    return (
        <div>
            <Navbar/>
            <h1>Video Player with Image</h1>
            <div style={{ display: 'flex' }}>
                <div style={{ marginLeft: '110px', marginTop:'125px' }}>
                    <h2 style={{marginLeft:'275px'}}>Video</h2>
                    <video controls width="640" height="360">
                        {/* <source src={videoSrc} type="video/mp4" /> */}
                        <source src='http://127.0.0.1:5000/getresultvideo/video' type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>
                <div style={{ marginLeft: '250px', marginTop:'125px' }}>
                    <h2 style={{marginLeft:'190px'}}>Image</h2>
                    <img src={imageSrc} alt="Server issuex" style={{ maxWidth: '640px', maxHeight: '360px' }} />
                </div>
            </div>
        </div>
    );
}

export default VideoPlayer;