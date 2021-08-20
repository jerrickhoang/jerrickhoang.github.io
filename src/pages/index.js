import React from 'react'

import GlobalWrapper from 'components/global-wrapper'
import Hero from 'components/hero'
import Section from 'components/section'
import About from 'components/about'
import Writing from 'components/writing'
import Footer from 'components/footer'
import { ParallaxProvider } from 'react-skrollr'
import Publication from '../components/publication'

const App = () => {
  return (
    <GlobalWrapper>
      {/* <Breakpoints /> */}
      <ParallaxProvider
        init={{
          smoothScrollingDuration: 500,
          smoothScrolling: true,
          forceHeight: false
        }}
      >
        <Hero />
        <Section id={'about'}>
          <About />
        </Section>
        <Section id={'projects'}>
          <Publication/>
        </Section>
        {/* <Section>
          <Writing />
        </Section> */}
      </ParallaxProvider>
      <Footer />
    </GlobalWrapper>
  )
}

export default App
