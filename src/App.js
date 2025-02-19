import Navbar from './components/Navbar';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import About from './components/About';
import Login from './components/Login';
import Register from './components/Register';
import VideoUpload from './components/Uploadvideo';
import VideoPlayer from './components/Videoplayer';
import Facialvarification from './components/Facialvarification';

function App() {
  return (
    <div>
      <Router>
        {/* <Navbar /> */}
        <Routes>
          <Route exact path="/home" element={<Home />}></Route>
          <Route exact path="/about" element={<About/>}></Route>
          <Route exact path="/fv" element={<Facialvarification/>}></Route>
          <Route exact path="/" element={<Login/>}></Route>
          <Route exact path="/register" element={<Register/>}></Route>
          <Route exact path="/upload" element={<VideoUpload/>}></Route>
          <Route exact path="/result" element={<VideoPlayer/>}></Route>
        </Routes>
      </Router>
    </div>
  );
}

export default App;
