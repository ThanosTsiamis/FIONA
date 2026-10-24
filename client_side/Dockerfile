FROM node:20-alpine

WORKDIR /usr/src/app
ENV PATH /usr/src/app/node_modules/.bin:$PATH

# install and cache app dependencies
COPY package.json /usr/src/app/package.json

RUN npm install

# next dev listens on 3000, and docker-compose publishes 3000:3000
EXPOSE 3000
CMD ["npm", "run", "dev"]
