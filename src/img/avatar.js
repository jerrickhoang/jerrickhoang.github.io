import React from 'react'

import styled from 'styled-components'


const Img = styled.img`
    border-radius: 50%;
    width: 100%;
`


const Avatar = () => {
    return (
        <Img src="./avatar.jpg">
        </Img>
    )
}

export default Avatar