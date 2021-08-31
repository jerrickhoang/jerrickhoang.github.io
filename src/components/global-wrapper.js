import React, { Fragment, useState, useEffect } from 'react'
import { Helmet } from 'react-helmet'
import GlobalStyle from 'styles/global-style'

const GlobalWrapper = (props) => {
  const [displayOutlines, setDisplayOutlines] = useState(false)

  const handleKeyboardInput = (e) => {
    const key = e.keyCode || e.charCode
    // Tab
    if (key === 9) {
      setDisplayOutlines(true)
    }
  }

  useEffect(() => {
    window.addEventListener('keydown', (e) => handleKeyboardInput(e))
  })

  return (
    <Fragment>
      <Helmet>
        <html lang="en" />
        <title>Jerrick Hoang</title>
        <meta name="description" content="Jerrick Hoang's projects" />
        <meta
          name="keywords"
          content="robotics, machine learning, autonomous, vehicles, cruise, waymo, cmu, uber, self driving, tesla, autopilot"
        />
        <meta property="og:description" content="Jerrick Hoang's projects" />
        <meta property="og:type" content="website" />

        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#ffffff" />
      </Helmet>
      <GlobalStyle displayOutlines={displayOutlines} />
      {props.children}
    </Fragment>
  )
}

export default GlobalWrapper
