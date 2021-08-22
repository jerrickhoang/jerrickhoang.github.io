import React, { Fragment } from 'react'
import Link from 'gatsby-link'
import styled from 'styled-components'
import media from 'utils/media-queries'

import { color, fontSize } from 'styles/theme'

import TwoColumns from 'components/twoColumns'
import SectionHeading from 'components/sectionHeading'
// import { Parallax } from 'react-skrollr'
import Avatar from '../img/avatar'




const Big = styled.span`
  font-size: ${fontSize.f6};
  color: ${color.grey900};
  font-weight: 700;
  letter-spacing: -0.4px;
  line-height: 1.35;
  ${media.lg`
    font-size: ${fontSize.f5};
    letter-spacing: -0.3px;
  `}
  ${media.sm`
    font-size: ${fontSize.f5};
  `}
`

const Div = styled.div`
  display: flex;
  flex-direction: column;
`

const About = () => {
  return (
    // <Parallax 
    //   data={{
    //     'data-center-center': 'opacity: 1;',
    //     'data-bottom-top': 'opacity: 0.5',
    //   }}
    // >
    <TwoColumns
      leftColumn={
        <Div>
          {/* <SectionHeading></SectionHeading> */}
          <Avatar></Avatar>
        </Div>
      }
      rightColumn={
        <Fragment>
          <Big>
            Hi. I'm Jerrick Hoang. My Vietnamese name is (Hoàng Tuấn Anh). 
          </Big>
          <p>
            I'm a ML/Robotics Engineer at <a href="https://www.getcruise.com/"> Cruise </a> working on the intersection of Prediction and Planning
          </p>
          <p>
            Previously, I hung out at the Machine Learning Department at CMU, specifically the 
            <a href="https://www.autonlab.org/"> Auton Lab </a> where I worked with <a href="https://www.cs.cmu.edu/~schneide/"> Prof. Jeff Schneider</a>
          </p>
          <p>
            Before that, I had a great pleasure of working with <a href="https://jimmylba.github.io/"> Prof. Jimmy Ba</a> at the
            <a href="https://vectorinstitute.ai/">Vector Institute </a>
          </p>
          <p style={{ marginBottom: 0 }}>
            You can also find me procastinating on <a href="https://www.chess.com/member/nsa_magic"> Chess.com </a> and 
            <a href="https://lichess.org/@/nsa_magic"> Lichess.org </a>
          </p>
        </Fragment>
      }
    />
    // </Parallax>

  )
}

export default About
