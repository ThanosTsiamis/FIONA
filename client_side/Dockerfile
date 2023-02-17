FROM node:18.13-alpine

WORKDIR /usr/src/app
ENV PATH /usr/src/app/node_modules/.bin:$PATH

# install and cache app dependencies
COPY package.json /usr/src/app/package.json

RUN npm install

EXPOSE 3008
CMD ["npm", "run", "dev"]
