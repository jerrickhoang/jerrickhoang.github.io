import React from 'react';
import styled from 'styled-components'

import Iframe from 'react-iframe'

const Div = styled.div`
  position: absolute;
  left: 0;
  bottom: 0;
  right: 0;
  width: 100%;
  height: 90vh;
//   overflow: hidden;
`

const Cloud = () => {
    const iframe_url = "https://sketchfab.com/models/e4506ab2ce7e43a69c04f715e0f0d55d/embed?autostart=1"
    // const iframe_url = "https://sketchfab.com/models/53226a76c68d4c8ab724d71ccc971630/embed?autostart=1&ui_theme=dark"
    return (
        <Div>
            <Iframe url={iframe_url}
                    width="100%"
                    height="90%"
                    loading="eager"/>

        </Div>
    )
}

export default Cloud