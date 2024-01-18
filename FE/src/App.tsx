import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import { Stack } from '@mui/system';
import WoundImage from './components/WoundImage/WoundImage';
import ScoreResult from './components/ScoreResult/ScoreResult';
import ImageInput from './components/ImageInput/ImageInput';
import { useState } from 'react';
import Footer from './components/Footer/Footer';
import getPerdanakusumaScore from './api/call/getPerdanakusumaScore';
import { PerdanakusumaScore } from './models/perdanakusuma';

interface ScorePair {
  left: string;
  right: string;
}

function App() {
  const [woundImage, setWoundImage] = useState<string>('');
  const [showScore, setShowScore] = useState<boolean>(false);

  const [perdanakusumaScore, setPerdanakusumaScore ] = useState<PerdanakusumaScore>({
    color : 0,
    color_desc : '',
    exudate : 0,
    exudate_desc : '',
    inflammation:0,
    inflammation_desc :''
  })

  const [severity, setSeverity] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const fileInputChangeHandler = (fileSrc: string) => {
    setWoundImage(fileSrc)
  }

  const imageSubmitHandler = (xRay: File) => {
    setLoading(true);
    getPerdanakusumaScore(xRay)
      .then(res => {
        setPerdanakusumaScore(res.data.data.predictions);
        setSeverity(res.data.data.status);
        setShowScore(true);
      })
      .catch(e => {
        setShowScore(false);
      })
      .finally(() => {
        setLoading(false)
      });
  }

  return (
    <>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar style={{ backgroundColor: 'whitesmoke'}}>
          <img src="/wound.png" alt="Wound Icon" style={{ marginRight: '10px' }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }} style={{ color : 'black'}}>              
              CHRONIC WOUND CLASSIFICATION
            </Typography>
          </Toolbar>
        </AppBar>
      </Box>
      <Box sx={{ minWidth: '200px', width: '60%', height: 'auto', margin: '4rem auto 0' }}>
        <Stack direction="row">
          <WoundImage imgUrl={woundImage} />
          <Stack sx={{ flex: 1, height: 'auto' }} direction="column" justifyContent="space-between">
            <Typography variant="h5" mb={2}>Result</Typography>
            <ScoreResult
              colorScore={perdanakusumaScore.color}
              colorDesc = {perdanakusumaScore.color_desc}
              exudateScore={perdanakusumaScore.exudate}
              exudateDesc = {perdanakusumaScore.exudate_desc}
              inflammationScore={perdanakusumaScore.inflammation}
              inflammationDesc={perdanakusumaScore.inflammation_desc}
              severity={severity}
              show={showScore}
            />
            <ImageInput
              onFileChange={fileInputChangeHandler}
              onSubmit={imageSubmitHandler}
              loading={loading}
            />
          </Stack>
        </Stack>
      </Box>
      <Footer />
    </>
  );
}

export default App
