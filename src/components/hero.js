import React from 'react'
import styled from 'styled-components'
import media from 'utils/media-queries'
import Cloud from 'components/cloud'
import Header from 'components/header'

const HeroSection = styled.section`
  height: 100vh;
  max-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  width: 100%;
  ${media.sm`
    height: calc(100vh - 76px);
  `}
`

const Hero = () => {
  return (
    <HeroSection>
      <Cloud />
      <Header />
    </HeroSection>
  )
}
export default Hero
