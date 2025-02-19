import axios from 'axios';
import React, { useState } from 'react';
// import { useHistory } from "react-router-dom";
import { Link, useNavigate } from 'react-router-dom';


function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [message, setMessage] = useState('');
    // const history = useHistory();
    const navigate = useNavigate();


    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://127.0.0.1:5000/login', {
                username: email,
                password: password
            });
            setMessage('Login successful');
            console.log(response.data); // You can handle the response here
            // this.props.history.push('/');
            navigate('/home');

        } catch (error) {
            console.error('Login error:', error);
            setMessage('Login failed. Please check your credentials.');
            // Handle error or show a message to the user
        }
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <div className='card' style={{ width: '25rem' }}>
                <div className='card-body'>
                    {/* {message && <p>{message}</p>} */}
                    <form onSubmit={handleLogin}>
                        <div className="mb-3">
                            <label htmlFor="exampleInputEmail1" className="form-label">Email address</label>
                            <input
                                type="email"
                                className="form-control"
                                id="exampleInputEmail1"
                                aria-describedby="emailHelp"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                            <div id="emailHelp" className="form-text">We'll never share your email with anyone else.</div>
                        </div>
                        <div className="mb-3">
                            <label htmlFor="exampleInputPassword1" className="form-label">Password</label>
                            <input
                                type="password"
                                className="form-control"
                                id="exampleInputPassword1"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>
                        <div className="mb-3 form-check">
                            <input type="checkbox" className="form-check-input" id="exampleCheck1" />
                            <label className="form-check-label" htmlFor="exampleCheck1">Check me out</label>
                        </div>
                        <button type="submit" className="btn btn-primary">Submit</button>
                        <Link to="/register" className="btn btn-dark">Register</Link>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default Login;
