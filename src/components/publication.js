import React, { Fragment } from 'react'
import Link from 'gatsby-link'

import TwoColumns from 'components/twoColumns'
import SectionHeading from 'components/sectionHeading'
import Project from 'components/project'

import MBRLLogo from 'img/mbrllogo'
import PPNetLogo from 'img/ppnetlogo'
import GoalNetLogo from 'img/goalnetlogo'
import TrafficGraphNetLogo from 'img/trafficgraphnetlogo'
import JointLogo from 'img/joint'
import PatentLogo from 'img/patentlogo'

import { Parallax } from 'react-skrollr'


const PPNetLink = <Link to="https://arxiv.org/abs/2104.14679">Read More</Link>
const TrafficGraphNetLink = <Link to="https://arxiv.org/abs/2009.12916">Read More</Link>
const MBRLLink = <Link to="https://arxiv.org/abs/1907.02057">Read More</Link>
const GoalNetLink = <Link to="https://arxiv.org/abs/2009.04450">Read More</Link>
const JointInteractionNetLink = <Link to="https://arxiv.org/abs/1912.07882">Read More</Link>


const Publication = () => {
  return (
    <Parallax 
      data={{
        'data-center-center': 'opacity: 1;',
        'data-bottom-top': 'opacity: 0.8',
      }}
    >
    <TwoColumns
      wide
      leftColumn={<SectionHeading>Publications</SectionHeading>}
      rightColumn={
        <Fragment>
          <Project
            logo={PPNetLogo()}
            title="Physically Feasible Vehicle Trajectory Prediction"
            abstract="2020 Conference on Neural Information Processing Systems (Neurips), ML4AD workshop "
            link={PPNetLink}
          />
          <Project
            logo={TrafficGraphNetLogo()}
            title="Interaction-Based Trajectory Prediction Over a Hybrid Traffic Graph"
            abstract="2021 International Conference on Intelligent Robots and Systems (IROS 2021)"
            link={TrafficGraphNetLink}
          />
          <Project
            logo={GoalNetLogo()}
            title="Map-Adaptive Goal-Based Trajectory Prediction"
            abstract="2020 International Conference on Robotics and Automation (ICRA 2020)"
            link={GoalNetLink}
          />
          <Project
            logo={JointLogo()}
            title="Joint interaction and trajectory prediction for autonomous driving using graph neural networks"
            abstract="2019 Conference on Neural Information Processing Systems (Neurips), ML4AD workshop "
            link={JointInteractionNetLink}
          />
          <Project
            logo={MBRLLogo()}
            title="Benchmarking model-based reinforcement learning"
            abstract="CoRR, vol. abs/1907.02057, 2019"
            link={MBRLLink}
          />
          <Project
            logo={PatentLogo()}
            title="Anomaly detection systems and methods for autonomous vehicles"
            abstract="US Patent UP-00466US"
          />
        </Fragment>
      }
    />
    </Parallax>
  )
}

export default Publication
